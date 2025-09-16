# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
Preliminary review of ACA catalogs selected by proseco.
"""

import gzip
import io
import pickle
import pprint
import re
import traceback
from pathlib import Path

import chandra_aca
import numpy as np
import proseco
from astropy.table import Table
from jinja2 import Template
from proseco.core import MetaAttribute

from sparkles import checks
from sparkles.aca_check_table import ACACheckTable
from sparkles.messages import Message, MessagesList
from sparkles.roll_optimize import RollOptimizeMixin

CACHE = {}
FILEDIR = Path(__file__).parent

# Minimum number of "anchor stars" that are always evaluated *without* the bonus
# from dynamic background when dyn_bgd_n_faint > 0. This is mostly to avoid the
# situation where 4 stars are selected and 2 are faint bonus stars. In this case
# there would be only 2 anchor stars that ensure good tracking even without
# dyn bgd.
MIN_DYN_BGD_ANCHOR_STARS = 3


def main(sys_args=None):
    """Command line interface to preview_load()"""
    import argparse

    from . import __version__

    parser = argparse.ArgumentParser(
        description=f"Sparkles ACA review tool {__version__}"
    )
    parser.add_argument(
        "load_name",
        type=str,
        help=(
            "Load name (e.g. JAN2119A) or full file name or directory "
            r"on \\noodle\GRETA\mission or OCCweb "
            r"containing a single pickle file, "
            r"or a Noodle path (beginning with \\noodle\GRETA\mission) "
            "or OCCweb path to a gzip pickle file"
        ),
    )
    parser.add_argument(
        "--obsid",
        action="append",
        type=float,
        help="Process only this obsid (can specify multiple times, default=all",
    )
    parser.add_argument("--quiet", action="store_true", help="Run quietly")
    parser.add_argument("--dyn-bgd-n-faint", type=int, help="Dynamic bgd bonus stars")
    parser.add_argument(
        "--open-html", action="store_true", help="Open HTML in webbrowser"
    )
    parser.add_argument(
        "--report-dir", type=str, help="Report output directory (default=<load name>"
    )
    parser.add_argument(
        "--report-level",
        type=str,
        default="none",
        help=(
            "Make reports for messages at/above level "
            "('all'|'none'|'info'|'caution'|'warning'|'critical') "
            "(default='warning')"
        ),
    )
    parser.add_argument(
        "--roll-level",
        type=str,
        default="none",
        help=(
            "Make alternate roll suggestions for messages at/above level "
            "('all'|'none'|'info'|'caution'|'warning'|'critical') "
            "(default='critical')"
        ),
    )
    parser.add_argument(
        "--roll-min-improvement",
        type=float,
        default=None,
        help="Min value of improvement metric to accept option (default=0.3)",
    )
    parser.add_argument(
        "--roll-d-roll",
        type=float,
        default=None,
        help=(
            "Delta roll for sampling available roll range "
            "(deg, default=0.25 for uniq_ids method and 0.5 for uniform method"
        ),
    )
    parser.add_argument(
        "--roll-max-roll-dev",
        type=float,
        default=None,
        help="maximum roll deviation (deg, default=max allowed by pitch)",
    )
    help = (
        "Method for determining roll intervals ('uniq_ids' | 'uniform'). "
        "The 'uniq_ids' method is a faster method that frequently finds an acceptable "
        "roll option, while 'uniform' is a brute-force search of the entire roll "
        "range at ``d_roll`` increments. If not provided, the default is to try "
        "*both* methods in order, stopping when an acceptable option is found."
    )
    parser.add_argument("--roll-method", type=str, default=None, help=help)
    parser.add_argument(
        "--roll-max-roll-options",
        type=int,
        default=None,
        help="maximum number of roll options to return (default=10)",
    )
    args = parser.parse_args(sys_args)

    # Parse the roll_args from command line args, only handling non-None values
    roll_args = {}
    for name in (
        "d_roll",
        "max_roll_dev",
        "min_improvement",
        "max_roll_options",
        "method",
    ):
        roll_name = "roll_" + name
        if getattr(args, roll_name) is not None:
            roll_args[name] = getattr(args, roll_name)
    if not roll_args:
        roll_args = None

    run_aca_review(
        args.load_name,
        report_dir=args.report_dir,
        report_level=args.report_level,
        roll_level=args.roll_level,
        loud=(not args.quiet),
        obsids=args.obsid,
        open_html=args.open_html,
        roll_args=roll_args,
        dyn_bgd_n_faint=args.dyn_bgd_n_faint,
    )


def run_aca_review(
    load_name=None,
    *,
    acars=None,
    make_html=True,
    report_dir=None,
    report_level="none",
    roll_level="none",
    roll_args=None,
    dyn_bgd_n_faint=None,
    loud=False,
    obsids=None,
    open_html=False,
    context=None,
    raise_exc=True,
):
    """Do ACA load review based on proseco pickle file from ORviewer.

    The ``load_name`` specifies the pickle file from which the ``ACATable``
    catalogs and obsids are read (unless ``acas`` and ``obsids`` are explicitly
    provided).  The following options are tried in this order:

    - <load_name> (e.g. 'JAN2119A_proseco.pkl')
    - <load_name>_proseco.pkl (for <load_name> like 'JAN2119A', ORviewer default)
    - <load_name>.pkl

    Instead of reading from a pickle, one can directly provide the catalogs as
    ``acars``.  In this case the ``load_name`` will only be used in the report
    HTML.

    When reading from a pickle, the ``obsids`` argument can be used to limit
    the list of obsids being processed.  This is handy for development or
    for examining just one obsid.

    If ``report_dir`` is not provided then it will be set to ``load_name``.

    The ``report_level`` arg specifies the message category at which the full
    HTML report for guide and acquisition will be generated for obsids with at
    least one message at or above that level.  The options correspond to
    standard categories "info", "caution", "warning", and "critical".  The
    default is "none", meaning no reports are generated.  A final option is
    "all" which generates a report for every obsid.

    The ``roll_level`` arg specifies the message category at which the review
    process will also attempt to find and report on available star catalogs for
    different roll options. The available categories are the same as for
    ``report_level``, with the most common choice being "critical".

    The ``roll_args`` arg specifies an optional dict of arguments used in the
    call to the ``get_roll_options`` method. Possible args include:

    - ``min_improvement``: min value of improvement metric to accept option
      (default=0.3)
    - ``d_roll``: delta roll for sampling available roll range (deg, default=0.25
        for uniq_ids method and 0.5 for uniform method)
    - ``max_roll_dev``: maximum roll deviation (deg, default=max allowed by pitch)
    - ``method``: method for determining roll intervals ('uniq_ids' | 'uniform').
      The 'uniq_ids' method is a faster method that frequently finds an acceptable
      roll option, while 'uniform' is a brute-force search of the entire roll
      range at ``d_roll`` increments. If not provided, the default is to try
      *both* methods in order, stopping when an acceptable option is found.
    - ``max_roll_options``: maximum number of roll options to return (default=10)

    If roll options are returned then they are sorted by the following keys:

    - Number of warnings at ``roll_level`` or worse (e.g. number of criticals)
      in ascending order (fewer is better)
    - Improvement in descending order (larger improvement is better)

    :param load_name: name of loads
    :param acars: list of ACAReviewTable objects (optional)
    :param make_html: make HTML output report
    :param open_html: open the HTML output in default web brower
    :param report_dir: output directory
    :param report_level: report level threshold for generating acq and guide report
    :param roll_level: level threshold for suggesting alternate rolls
    :param roll_args: None or dict of arguments for ``get_roll_options``
    :param dyn_bgd_n_faint: int for dynamic background bonus calculation.  Overrides value
        in input.
    :param loud: print status information during checking
    :param obsids: list of obsids for selecting a subset for review (mostly for debug)
    :param is_ORs: list of is_OR values (for roll options review page)
    :param context: initial context dict for HTML report
    :param raise_exc: if False then catch exception and return traceback (default=True)
    :returns: exception message: str or None

    """
    try:
        _run_aca_review(
            load_name=load_name,
            acars=acars,
            make_html=make_html,
            report_dir=report_dir,
            report_level=report_level,
            roll_level=roll_level,
            roll_args=roll_args,
            dyn_bgd_n_faint=dyn_bgd_n_faint,
            loud=loud,
            obsids=obsids,
            open_html=open_html,
            context=context,
        )
    except Exception:
        if raise_exc:
            raise
        exception = traceback.format_exc()
    else:
        exception = None

    return exception


def _run_aca_review(
    load_name=None,
    *,
    acars=None,
    make_html=True,
    report_dir=None,
    report_level="none",
    roll_level="none",
    roll_args=None,
    dyn_bgd_n_faint=None,
    loud=False,
    obsids=None,
    open_html=False,
    context=None,
):
    if acars is None:
        acars, load_name = get_acas_from_pickle(load_name, loud)

    if obsids:
        acars = [aca for aca in acars if aca.obsid in obsids]

    if not acars:
        raise ValueError("no catalogs founds (check obsid filtering?)")

    if roll_args is None:
        roll_args = {}

    # Make output directory if needed
    if make_html:
        # Generate outdir from load_name if necessary
        if report_dir is None:
            if not load_name:
                raise ValueError(
                    "load_name must be provided if outdir is not specified"
                )
            # Chop any directory path from load_name
            load_name = Path(load_name).name
            report_dir = re.sub(r"(_proseco)?.pkl(.gz)?", "", load_name) + "_sparkles"
        report_dir = Path(report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)

    # Do the sparkles review for all the catalogs
    for aca in acars:
        if not isinstance(aca, ACAReviewTable):
            raise TypeError("input catalog for review must be an ACAReviewTable")

        if loud:
            print(f"Processing obsid {aca.obsid}")

        aca.messages.clear()
        aca.context.clear()

        aca.set_stars_and_mask()  # Not stored in pickle, need manual restoration

        if dyn_bgd_n_faint is not None:
            if aca.dyn_bgd_n_faint != dyn_bgd_n_faint:
                aca.add_message(
                    "info",
                    text=(
                        f"Using dyn_bgd_n_faint={dyn_bgd_n_faint} "
                        f"(call_args val={aca.dyn_bgd_n_faint})"
                    ),
                )
                aca.dyn_bgd_n_faint = dyn_bgd_n_faint
                aca.guides.dyn_bgd_n_faint = dyn_bgd_n_faint

        check_catalog(aca)

        # Find roll options if requested
        if roll_level == "all" or aca.messages >= roll_level:
            # Get roll selection algorithms to try
            max_roll_options = roll_args.pop("max_roll_options", 10)
            methods = roll_args.pop(
                "method", ("uniq_ids", "uniform") if aca.is_OR else "uniq_ids"
            )
            if isinstance(methods, str):
                methods = [methods]

            try:
                # Set roll_options, roll_info attributes
                for method in methods:
                    aca.roll_options = None
                    aca.roll_info = None
                    aca.get_roll_options(method=method, **roll_args)
                    aca.roll_info["method"] = method

                    # If there is at least one option with no messages at the
                    # roll_level (typically "critical") then declare success and
                    # stop looking for roll options.
                    if any(
                        not roll_option["acar"].messages >= roll_level
                        for roll_option in aca.roll_options
                    ):
                        break

                aca.sort_and_limit_roll_options(roll_level, max_roll_options)

            except Exception:  # as err:
                err = traceback.format_exc()
                aca.add_message(
                    "critical", text=f"Running get_roll_options() failed: \n{err}"
                )
                aca.roll_options = None
                aca.roll_info = None

        if make_html:
            # Output directory for the main prelim review index.html and for this obsid.
            # Note that the obs{aca.obsid} is not flexible because it must match the
            # convention used in ACATable.make_report().  Oops.
            aca.preview_dir = Path(report_dir)
            aca.obsid_dir = aca.preview_dir / f"obs{aca.obsid:05}"
            aca.obsid_dir.mkdir(parents=True, exist_ok=True)

            aca.make_starcat_plot()

            if report_level == "all" or aca.messages >= report_level:
                try:
                    aca.make_report()
                except Exception:
                    err = traceback.format_exc()
                    aca.add_message(
                        "critical", text=f"Running make_report() failed:\n{err}"
                    )

            if aca.roll_info:
                aca.make_roll_options_report()

            aca.context["text_pre"] = aca.get_text_pre()
            aca.context["call_args"] = aca.get_call_args_pre()

    # noinspection PyDictCreation
    if make_html:
        from . import __version__

        # Create new context or else use a copy of the supplied dict
        if context is None:
            context = {}
        else:
            context = context.copy()

        context["load_name"] = load_name.upper()
        context["proseco_version"] = proseco.__version__
        context["sparkles_version"] = __version__
        context["chandra_aca_version"] = chandra_aca.__version__
        context["acas"] = acars
        context["summary_text"] = get_summary_text(acars)

        # Special case when running a set of roll options for one obsid
        is_roll_report = all(aca.is_roll_option for aca in acars)

        label_frame = "ACA" if aca.is_ER else "Target"
        context["id_label"] = f"{label_frame} roll" if is_roll_report else "Obsid"

        template_file = FILEDIR / "index_template_preview.html"
        template = Template(open(template_file, "r").read())
        out_html = template.render(context)

        out_filename = report_dir / "index.html"
        if loud:
            print(f"Writing output review file {out_filename}")
        with open(out_filename, "w") as fh:
            fh.write(out_html)

        if open_html:
            import webbrowser

            out_url = f"file://{out_filename.absolute()}"
            print(f"Open URL in browser: {out_url}")
            webbrowser.open(out_url)


def stylize(text, category):
    """Stylize ``text``.

    Currently ``category`` of critical, warning, caution, or info are supported
    in the CSS span style.

    """
    out = f'<span class="{category}">{text}</span>'
    return out


def get_acas_dict_from_local_file(load_name, loud):
    filenames = [
        load_name,
        f"{load_name}_proseco.pkl.gz",
        f"{load_name}.pkl.gz",
        f"{load_name}_proseco.pkl",
        f"{load_name}.pkl",
    ]
    for filename in filenames:
        pth = Path(filename)
        if (
            pth.exists()
            and pth.is_file()
            and pth.suffixes in ([".pkl"], [".pkl", ".gz"])
        ):
            if loud:
                print(f"Reading pickle file {filename}")

            open_func = open if pth.suffix == ".pkl" else gzip.open
            with open_func(filename, "rb") as fh:
                acas_dict = pickle.load(fh)

            break
    else:
        raise FileNotFoundError(f"no matching pickle file {filenames}")
    return acas_dict, pth.name


def get_acas_dict_from_occweb(path):
    """Get pickle file from OCCweb"""
    from kadi.occweb import get_occweb_dir, get_occweb_page

    if not path.endswith(".pkl.gz"):
        occweb_files = get_occweb_dir(path)
        pkl_files = [name for name in occweb_files["Name"] if name.endswith(".pkl.gz")]
        if len(pkl_files) > 1:
            print(f"Found multiple pickle files for {path}:")
            for i, pkl_file in enumerate(pkl_files):
                print(f"  {i}: {pkl_file}")
            choice = input("Enter choice: ")
            pkl_files = [pkl_files[int(choice)]]
        elif len(pkl_files) == 0:
            raise ValueError(f"No pickle files found in {path}")
        path = path + "/" + pkl_files[0]

    content = get_occweb_page(path, binary=True, cache=True)
    acas_dict = pickle.loads(gzip.decompress(content))
    return acas_dict, Path(path).name


def get_acas_from_pickle(load_name, loud=False):
    r"""Get dict of proseco ACATable pickles for ``load_name``

    ``load_name`` can be a full file name (ending in .pkl or .pkl.gz) or any of the
     following, which are tried in order:

    - <load_name>_proseco.pkl.gz
    - <load_name>.pkl.gz
    - <load_name>_proseco.pkl
    - <load_name>.pkl

    ``load_name`` can also be a directory on Noodle or OCCweb containing a exactly
    one pickle file, or a Noodle path or OCCweb path to a gzip pickle file:

    - \\noodle\GRETA\mission\<path-to-pickle-file>\  (looks for one *.pkl.gz file)
    - \\noodle\GRETA\mission\<path-to-pickle-file>\<load_name>_proseco.pkl.gz
    - https://occweb.cfa.harvard.edu/occweb/<path-to-pickle-file>\  (looks for one *.pkl.gz file)
    - https://occweb.cfa.harvard.edu/occweb/<path-to-pickle-file>\<load_name>_proseco.pkl.gz

    :param load_name: load name
    :param loud: print processing information
    """
    if load_name.startswith((r"\\noodle", "https://occweb")):
        acas_dict, path_name = get_acas_dict_from_occweb(load_name)
    else:
        acas_dict, path_name = get_acas_dict_from_local_file(load_name, loud)

    acas = [
        ACAReviewTable(aca, obsid=obsid, loud=loud) for obsid, aca in acas_dict.items()
    ]
    return acas, path_name


def get_summary_text(acas):
    """Get summary text for all catalogs.

    This is like::

      Proseco version: 4.4-r528-e9d6c73

      OBSID = -3898   at 2019:027:21:58:37.828   7.8 ACQ | 5.0 GUI |
      OBSID = 21899   at 2019:028:01:17:39.066   7.8 ACQ | 5.0 GUI |

    :param acas: list of ACATable objects
    :returns: str summary text
    """
    report_id_strs = [str(aca.report_id) for aca in acas]
    max_obsid_len = max(len(obsid_str) for obsid_str in report_id_strs)
    lines = []
    for aca, report_id_str in zip(acas, report_id_strs):
        fill = " " * (max_obsid_len - len(report_id_str))
        # Is this being generated for a roll options report?
        ident = "ROLL" if aca.is_roll_option else "OBSID"
        line = (
            f'<a href="#id{aca.report_id}">{ident} = {report_id_str}</a>{fill}'
            f" at {aca.date}   "
            f"{aca.acq_count:.1f} ACQ | {aca.guide_count:.1f} GUI |"
        )

        # Warnings
        for category in reversed(MessagesList.categories):
            msgs = aca.messages == category
            if msgs:
                text = stylize(f" {category.capitalize()}: {len(msgs)}", category)
                line += text

        lines.append(line)

    return "\n".join(lines)


class ACAReviewTable(ACACheckTable, RollOptimizeMixin):
    # Whether this instance is a roll option (controls how HTML report page is formatted)
    is_roll_option = MetaAttribute()
    roll_options = MetaAttribute()
    roll_info = MetaAttribute()

    def __init__(self, *args, **kwargs):
        """Init review methods and attrs in ``aca`` object *in-place*.

        - Change ``aca.__class__`` to ``cls``
        - Add ``context`` and ``messages`` properties.

        :param aca: ACATable object, modified in place.
        :param obsid: obsid (optional)
        :param loud: print processing status info (default=False)

        """
        # if data is None:
        #    raise ValueError(f'data arg must be set to initialize {self.__class__.__name__}')

        loud = kwargs.pop("loud", False)
        is_roll_option = kwargs.pop("is_roll_option", False)

        # Make a copy of input aca table along with a deepcopy of its meta.
        super().__init__(*args, **kwargs)

        # If no data were provided then skip all the rest of the initialization.
        # This happens during slicing. The result is not actually
        # a functional ACAReviewTable, but it allows inspection of data.
        if len(self.colnames) == 0:
            return

        # Init roll option attrs. Note that an instance might be initialized from
        # another acar instance that already has these attrs set, so we need to
        # explicitly set them here.
        self.is_roll_option = is_roll_option
        self.roll_options = None
        self.roll_info = None

        # Instance attributes that won't survive pickling
        self.context = {}  # Jinja2 context for output HTML review
        self.loud = loud
        self.preview_dir = None
        self.obsid_dir = None
        self.roll_options_table = None

    def run_aca_review(
        self,
        *,
        make_html=False,
        report_dir=".",
        report_level="none",
        roll_level="none",
        roll_args=None,
        raise_exc=True,
    ):
        """Do aca review based for this catalog

        The ``report_level`` arg specifies the message category at which the full
        HTML report for guide and acquisition will be generated for obsids with at
        least one message at or above that level.  The options correspond to
        standard categories "info", "caution", "warning", and "critical".  The
        default is "none", meaning no reports are generated.  A final option is
        "all" which generates a report for every obsid.

        The ``roll_level`` arg specifies the message category at which the review
        process will also attempt to find and report on available star catalogs for
        different roll options. The available categories are the same as for
        ``report_level``, with the most common choice being "critical".

        The ``roll_args`` arg specifies an optional dict of arguments used in the
        call to the ``get_roll_options`` method. Possible args include:

        - ``min_improvement``: min value of improvement metric to accept option
          (default=0.3)
        - ``d_roll``: delta roll for sampling available roll range (deg, default=0.25
            for uniq_ids method and 0.5 for uniform method)
        - ``max_roll_dev``: maximum roll deviation (deg, default=max allowed by pitch)
        - ``method``: method for determining roll intervals ('uniq_ids' | 'uniform').
          The 'uniq_ids' method is a faster method that frequently finds an acceptable
          roll option, while 'uniform' is a brute-force search of the entire roll
          range at ``d_roll`` increments. If not provided, the default is to try
          *both* methods in order, stopping when an acceptable option is found.

        :param make_html: make HTML report (default=False)
        :param report_dir: output directory for report (default='.')
        :param report_level: report level threshold for generating acq and guide report
        :param roll_level: level threshold for suggesting alternate rolls
        :param roll_args: None or dict of arguments for ``get_roll_options``
        :param raise_exc: if False then catch exception and return traceback (default=True)
        :returns: exception message: str or None

        """
        acars = [self]

        # Do aca review checks and update acas[0] in place
        exc = run_aca_review(
            load_name=f"Obsid {self.obsid}",
            acars=acars,
            make_html=make_html,
            report_dir=report_dir,
            report_level=report_level,
            roll_level=roll_level,
            roll_args=roll_args,
            loud=False,
            raise_exc=raise_exc,
        )
        return exc

    def review_status(self):
        if self.thumbs_up:
            status = 1
        elif self.thumbs_down:
            status = -1
        else:
            status = 0

        return status

    @property
    def report_id(self):
        return round(self.att_targ.roll, 2) if self.is_roll_option else self.obsid

    @property
    def thumbs_up(self):
        n_crit_warn = len(self.messages == "critical") + len(self.messages == "warning")
        return n_crit_warn == 0

    @property
    def thumbs_down(self):
        n_crit = len(self.messages == "critical")
        return n_crit > 0

    def make_report(self):
        """Make report for acq and guide."""
        if self.loud:
            print(f"  Creating HTML reports for obsid {self.obsid}")

        # Make reports in preview_dir/obs<obsid>/{acq,guide}/
        super().make_report(rootdir=self.preview_dir)

        # Let the jinja template know this has reports and set the correct
        # relative link from <preview_dir>/index.html to the reports directory
        # containing acq/index.html and guide/index.html files.
        self.context["reports_dir"] = self.obsid_dir.relative_to(
            self.preview_dir
        ).as_posix()

    def set_stars_and_mask(self):
        """Set stars attribute for plotting.

        This includes compatibility code to deal with somewhat-broken behavior
        in 4.3.x where the base plot method is hard-coded to look at
        ``acqs.stars`` and ``acqs.bad_stars``.

        """
        # Get stars from AGASC and set ``stars`` attribute
        self.set_stars(filter_near_fov=False)

    def get_roll_options_table(self):
        """Return table of roll options

        Note that self.roll_options includes the originally-planned roll case
        as the first row.

        """
        opts = [opt.copy() for opt in self.roll_options]

        for opt in opts:
            del opt["acar"]
            for name in ("add_ids", "drop_ids"):
                opt[name] = (
                    " ".join(str(id_) for id_ in opt[name]) if opt[name] else "--"
                )

        opts_table = Table(
            opts,
            names=[
                "roll",
                "P2",
                "n_stars",
                "improvement",
                "roll_min",
                "roll_max",
                "add_ids",
                "drop_ids",
            ],
        )
        for col in opts_table.itercols():
            if col.dtype.kind == "f":
                col.info.format = ".2f"

        return opts_table

    def make_roll_options_report(self):
        """Make a summary table and separate report page for roll options."""
        self.roll_options_table = self.get_roll_options_table()

        # rolls = [opt['acar'].att.roll for opt in self.roll_options]
        acas = [opt["acar"] for opt in self.roll_options]

        # Add in a column with summary of messages in roll options e.g.
        # critical: 2 warning: 1
        msgs_summaries = []
        for aca in acas:
            texts = []
            for category in reversed(MessagesList.categories):
                msgs = aca.messages == category
                if msgs:
                    text = stylize(f"{category.capitalize()}: {len(msgs)}", category)
                    texts.append(text)
            msgs_summaries.append(" ".join(texts))
        self.roll_options_table["warnings"] = msgs_summaries

        # Set context for HTML output
        rolls_index = (
            self.obsid_dir.relative_to(self.preview_dir) / "rolls" / "index.html"
        )
        io_html = io.StringIO()
        self.roll_options_table.write(
            io_html,
            format="ascii.html",
            htmldict={
                "table_class": "table-striped",
                "raw_html_cols": ["warnings"],
                "raw_html_clean_kwargs": {"tags": ["span"], "attributes": ["class"]},
            },
        )
        htmls = [line.strip() for line in io_html.getvalue().splitlines()]
        htmls = htmls[
            htmls.index('<table class="table-striped">') : htmls.index("</table>") + 1
        ]
        roll_context = {}
        roll_context["roll_options_table"] = "\n".join(htmls)
        roll_context["roll_options_index"] = rolls_index.as_posix()
        for key in ("roll_min", "roll_max", "roll_nom"):
            roll_context[key] = f"{self.roll_info[key]:.2f}"
        roll_context["roll_method"] = self.roll_info["method"]
        self.context.update(roll_context)

        # Make a separate preview page for the roll options
        rolls_dir = self.obsid_dir / "rolls"
        run_aca_review(
            f"Obsid {self.obsid} roll options",
            acars=acas,
            report_dir=rolls_dir,
            report_level="none",
            roll_level="none",
            loud=False,
            context=roll_context,
        )

    def plot(self, ax=None, **kwargs):
        """
        Plot the catalog and background stars.

        :param ax: matplotlib axes object for plotting to (optional)
        :param kwargs: other keyword args for plot_stars
        """
        from matplotlib.patches import Circle

        fig = super().plot(ax, **kwargs)
        if ax is None:
            ax = fig.gca()

        # Increase plot bounds to allow seeing stars within a 750 pixel radius
        ax.set_xlim(-770, 770)  # pixels
        ax.set_ylim(-770, 770)  # pixels

        # Draw a circle at 735 pixels showing extent of CCD corners
        circle = Circle(
            (0, 0), radius=735, facecolor="none", edgecolor="g", alpha=0.5, lw=3
        )
        ax.add_patch(circle)

        # Plot a circle around stars that were not already candidates
        # for BOTH guide and acq, and were not selected as EITHER guide
        # or acq.  Visually this means highlighting new possibilities.
        idxs = self.get_candidate_better_stars()
        stars = self.stars[idxs]
        for star in stars:
            already_checked = (star["id"] in self.acqs.cand_acqs["id"]) and (
                star["id"] in self.guides.cand_guides["id"]
            )
            selected = star["id"] in set(self.acqs["id"]) | set(self.guides["id"])
            if not already_checked and not selected:
                circle = Circle(
                    (star["row"], star["col"]),
                    radius=20,
                    facecolor="none",
                    edgecolor="r",
                    alpha=0.8,
                    lw=1.5,
                )
                ax.add_patch(circle)
                ax.text(
                    star["row"] + 24,
                    star["col"],
                    f"{star['mag']:.2f}",
                    ha="left",
                    va="center",
                    fontsize="small",
                    color="r",
                )

    def make_starcat_plot(self):
        """Make star catalog plot for this observation."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plotname = f"starcat{self.report_id}.png"
        outfile = self.obsid_dir / plotname
        self.context["catalog_plot"] = outfile.relative_to(self.preview_dir).as_posix()

        fig = plt.figure(figsize=(6.75, 6.0))
        ax = fig.add_subplot(1, 1, 1)
        self.plot(ax=ax)
        plt.tight_layout()
        fig.savefig(str(outfile))
        plt.close(fig)

    def get_call_args_pre(self):
        call_args = self.call_args.copy()
        try:
            call_args["att"] = call_args["att"].q.tolist()
        except AttributeError:
            # Wasn't a quaternion, no action required.
            pass
        call_args["include_ids_acq"] = self.acqs["id"].tolist()
        call_args["include_halfws_acq"] = self.acqs["halfw"].tolist()
        call_args["include_ids_guide"] = self.guides["id"].tolist()
        out = pprint.pformat(call_args, width=120)
        return out

    def get_text_pre(self):
        """Get pre-formatted text for report."""
        P2 = -np.log10(self.acqs.calc_p_safe())
        att = self.att
        att_targ = self.att_targ
        self._base_repr_()  # Hack to set default ``format`` for cols as needed
        catalog = "\n".join(self.pformat(max_width=-1, max_lines=-1))

        att_string = (
            f"ACA RA, Dec, Roll (deg): {att.ra:.5f} {att.dec:.5f} {att.roll:.5f}"
        )
        if self.is_OR:
            att_string += (
                f"  [Target: {att_targ.ra:.5f} {att_targ.dec:.5f} {att_targ.roll:.5f}]"
            )

        message_text = self.get_formatted_messages()

        # Compare effective temperature (used in selection and evaluation process)
        # with actual predicted temperature.  If different then this observation
        # is close to the planning limit and has had a temperature penalty
        # applied during selection.  Define message(s) to include in temperature line(s).
        if np.isclose(self.t_ccd_guide, self.t_ccd_eff_guide):
            t_ccd_eff_guide_msg = ""
        else:
            t_ccd_eff_guide_msg = " " + stylize(
                f"(Effective : {self.t_ccd_eff_guide:.1f})", "caution"
            )

        if np.isclose(self.t_ccd_acq, self.t_ccd_eff_acq):
            t_ccd_eff_acq_msg = ""
        else:
            t_ccd_eff_acq_msg = " " + stylize(
                f"(Effective : {self.t_ccd_eff_acq:.1f})", "caution"
            )

        text_pre = f"""\
{self.detector} SIM-Z offset: {self.sim_offset}
{att_string}
Dither acq: Y_amp= {self.dither_acq.y:.1f}  Z_amp={self.dither_acq.z:.1f}
Dither gui: Y_amp= {self.dither_guide.y:.1f}  Z_amp={self.dither_guide.z:.1f}
Maneuver Angle: {self.man_angle:.2f}
Next Maneuver Angle: {self.man_angle_next:.2f}
Date: {self.date}

{catalog}

{message_text}\
Probability of acquiring 2 or fewer stars (10^-x): {P2:.2f}
Acquisition Stars Expected: {self.acq_count:.2f}
Guide Stars count: {self.guide_count:.2f}
Predicted Guide CCD temperature (max): {self.t_ccd_guide:.1f}{t_ccd_eff_guide_msg}
Predicted Acq CCD temperature (init) : {self.t_ccd_acq:.1f}{t_ccd_eff_acq_msg}"""

        return text_pre

    def get_formatted_messages(self):
        """Format message dicts into pre-formatted lines for the preview report."""
        lines = []
        for message in self.messages:
            category = message["category"]
            idx_str = f"[{message['idx']}] " if ("idx" in message) else ""
            line = f">> {category.upper()}: {idx_str}{message['text']}"
            line = stylize(line, category)
            lines.append(line)

        out = "\n".join(lines) + "\n\n" if lines else ""
        return out

    def add_message(self, category, text, **kwargs):
        r"""Add message to internal messages list.

        :param category: message category ('info', 'caution', 'warning', 'critical')
        :param text: message text
        :param \**kwargs: other kwarg

        """
        message = {"text": text, "category": category}
        message.update(kwargs)
        self.messages.append(message)

    @classmethod
    def from_ocat(cls, obsid, t_ccd=-5, man_angle=5, date=None, roll=None, **kwargs):
        """Return an AcaReviewTable object using OCAT to specify key information.

        :param obsid: obsid
        :param t_ccd: ACA CCD temperature (degrees C)
        :param man_angle: maneuver angle (degrees)
        :param date: observation date (for proper motion and ACA offset projection)
        :param roll: roll angle (degrees)
        :param **kwargs: additional keyword args to update or override params from
            yoshi for call to get_aca_catalog()

        :returns: AcaReviewTable object
        """
        from proseco import get_aca_catalog

        from .yoshi import convert_yoshi_to_proseco_params, get_yoshi_params_from_ocat

        params_yoshi = get_yoshi_params_from_ocat(obsid, obs_date=date)
        if roll is not None:
            params_yoshi["roll_targ"] = roll
        params_proseco = convert_yoshi_to_proseco_params(
            **params_yoshi, obsid=obsid, man_angle=man_angle, t_ccd=t_ccd
        )
        params_proseco.update(kwargs)
        aca = get_aca_catalog(**params_proseco)
        acar = cls(aca)
        return acar


check_acq_p2 = checks.acar_check_wrapper(checks.check_acq_p2)
check_bad_stars = checks.acar_check_wrapper(checks.check_bad_stars)
check_dither = checks.acar_check_wrapper(checks.check_dither)
check_config_for_no_guide_dither = checks.acar_check_wrapper(
    checks.check_config_for_no_guide_dither
)
check_fid_count = checks.acar_check_wrapper(checks.check_fid_count)
check_fid_spoiler_score = checks.acar_check_wrapper(checks.check_fid_spoiler_score)
check_guide_count = checks.acar_check_wrapper(checks.check_guide_count)
check_guide_fid_position_on_ccd = checks.acar_check_wrapper(
    checks.check_guide_fid_position_on_ccd
)
check_guide_geometry = checks.acar_check_wrapper(checks.check_guide_geometry)
check_run_jupiter_checks = checks.acar_check_wrapper(checks.check_run_jupiter_checks)
check_jupiter_acq_spoilers = checks.acar_check_wrapper(
    checks.check_jupiter_acq_spoilers
)
check_jupiter_track_spoilers = checks.acar_check_wrapper(
    checks.check_jupiter_track_spoilers
)
check_jupiter_distribution = checks.acar_check_wrapper(
    checks.check_jupiter_distribution
)
check_guide_is_candidate = checks.acar_check_wrapper(checks.check_guide_is_candidate)
check_guide_overlap = checks.acar_check_wrapper(checks.check_guide_overlap)
check_imposters_guide = checks.acar_check_wrapper(checks.check_imposters_guide)
check_include_exclude = checks.acar_check_wrapper(checks.check_include_exclude)
check_pos_err_guide = checks.acar_check_wrapper(checks.check_pos_err_guide)
check_too_bright_guide = checks.acar_check_wrapper(checks.check_too_bright_guide)


def check_catalog(acar: ACACheckTable) -> None:
    """Perform all star catalog checks."""
    msgs: list[Message] = []
    for entry in acar:
        entry_type = entry["type"]
        is_guide = entry_type in ("BOT", "GUI")
        is_acq = entry_type in ("BOT", "ACQ")
        is_fid = entry_type == "FID"

        if is_guide or is_fid:
            msgs += checks.check_guide_fid_position_on_ccd(acar, entry)

        if is_guide:
            star = acar.guides.get_id(entry["id"])
            msgs += checks.check_pos_err_guide(acar, star)
            msgs += checks.check_imposters_guide(acar, star)
            msgs += checks.check_too_bright_guide(acar, star)
            msgs += checks.check_guide_is_candidate(acar, star)

        if is_guide or is_acq:
            msgs += checks.check_bad_stars(entry)

        if is_fid:
            fid = acar.fids.get_id(entry["id"])
            msgs += checks.check_fid_spoiler_score(entry["idx"], fid)

    msgs += checks.check_guide_overlap(acar)

    # If the target_name includes "jupiter" then run jupiter checks.
    if acar.target_name is not None and "jupiter" in acar.target_name.lower():
        msgs += checks.check_run_jupiter_checks(acar)

    msgs += checks.check_guide_geometry(acar)

    msgs += checks.check_acq_p2(acar)
    msgs += checks.check_guide_count(acar)
    msgs += checks.check_dither(acar)
    msgs += checks.check_config_for_no_guide_dither(acar)
    msgs += checks.check_fid_count(acar)
    msgs += checks.check_include_exclude(acar)

    messages = [
        {
            key: val
            for key in ("category", "text", "idx")
            if (val := getattr(msg, key)) is not None
        }
        for msg in msgs
    ]

    acar.messages.extend(messages)
