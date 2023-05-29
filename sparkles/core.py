# coding: utf-8
# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
Preliminary review of ACA catalogs selected by proseco.
"""
import gzip
import io
import re
import traceback
from pathlib import Path
import pickle
from itertools import combinations, chain
import pprint

import matplotlib

matplotlib.use("Agg")  # noqa
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import numpy as np
from jinja2 import Template
from chandra_aca.star_probs import guide_count
from chandra_aca.transform import yagzag_to_pixels, mag_to_count_rate, snr_mag_for_t_ccd
import chandra_aca
from astropy.table import Column, Table

import proseco
from proseco.catalog import ACATable
import proseco.characteristics as ACA
from proseco.core import MetaAttribute

from .roll_optimize import RollOptimizeMixin


CACHE = {}
FILEDIR = Path(__file__).parent

# Minimum number of "anchor stars" that are always evaluated *without* the bonus
# from dynamic background when dyn_bgd_n_faint > 0. This is mostly to avoid the
# situation where 4 stars are selected and 2 are faint bonus stars. In this case
# there would be only 2 anchor stars that ensure good tracking even without
# dyn bgd.
MIN_DYN_BGD_ANCHOR_STARS = 3

# Observations with man_angle_next less than or equal to CREEP_AWAY_THRESHOLD
# are considered "creep away" observations. CREEP_AWAY_THRESHOLD is in units of degrees.
CREEP_AWAY_THRESHOLD = 5.0


def main(sys_args=None):
    """Command line interface to preview_load()"""
    from . import __version__

    import argparse

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
                    text=f"Using dyn_bgd_n_faint={dyn_bgd_n_faint} "
                         f"(call_args val={aca.dyn_bgd_n_faint})",
                )
                aca.dyn_bgd_n_faint = dyn_bgd_n_faint

        aca.check_catalog()

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
            aca.obsid_dir = aca.preview_dir / f"obs{aca.obsid}"
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
    if load_name.startswith(r"\\noodle") or load_name.startswith("https://occweb"):
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


class MessagesList(list):
    categories = ("all", "info", "caution", "warning", "critical", "none")

    def __eq__(self, other):
        if isinstance(other, str):
            return [msg for msg in self if msg["category"] == other]
        else:
            return super().__eq__(other)

    def __ge__(self, other):
        if isinstance(other, str):
            other_idx = self.categories.index(other)
            return [
                msg
                for msg in self
                if self.categories.index(msg["category"]) >= other_idx
            ]
        else:
            return super().__ge__(other)


class ACAReviewTable(ACATable, RollOptimizeMixin):
    # Whether this instance is a roll option (controls how HTML report page is formatted)
    is_roll_option = MetaAttribute()
    roll_options = MetaAttribute()
    roll_info = MetaAttribute()
    messages = MetaAttribute()

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

        obsid = kwargs.pop("obsid", None)
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

        # Add row and col columns from yag/zag, if not already there.
        self.add_row_col()

        self.messages = MessagesList()  # Warning messages

        # Instance attributes that won't survive pickling
        self.context = {}  # Jinja2 context for output HTML review
        self.loud = loud
        self.preview_dir = None
        self.obsid_dir = None
        self.roll_options_table = None

        # Input obsid could be a string repr of a number that might have have
        # up to 2 decimal points.  This is the case when obsid is taken from the
        # ORviewer dict of ACATable pickles from prelim review.  Tidy things up
        # in these cases.
        if obsid is not None:
            f_obsid = round(float(obsid), 2)
            i_obsid = int(f_obsid)
            num_obsid = i_obsid if (i_obsid == f_obsid) else f_obsid

            self.obsid = num_obsid
            self.acqs.obsid = num_obsid
            self.guides.obsid = num_obsid
            self.fids.obsid = num_obsid

        if (
            "mag_err" not in self.colnames
            and self.acqs is not None
            and self.guides is not None
        ):
            # Add 'mag_err' column after 'mag' using 'mag_err' from guides and acqs
            mag_errs = {
                entry["id"]: entry["mag_err"] for entry in chain(self.acqs, self.guides)
            }
            mag_errs = Column(
                [mag_errs.get(id, 0.0) for id in self["id"]], name="mag_err"
            )
            self.add_column(mag_errs, index=self.colnames.index("mag") + 1)

        # Don't want maxmag column
        if "maxmag" in self.colnames:
            del self["maxmag"]

        # Customizations for ACAReviewTable.  Don't really need 2 decimals on these.
        for name in ("yang", "zang", "row", "col"):
            self._default_formats[name] = ".1f"

        # Make mag column have an extra space for catalogs with all mags < 10.0
        self._default_formats["mag"] = "5.2f"

        if self.colnames[0] != "idx":
            # Move 'idx' to first column.  This is really painful currently.
            self.add_column(Column(self["idx"], name="idx_temp"), index=0)
            del self["idx"]
            self.rename_column("idx_temp", "idx")

    @property
    def t_ccds_bonus(self):
        """Effective T_ccd for each guide star, including dynamic background bonus."""
        if not hasattr(self, "_t_ccds_bonus"):
            # Compute guide count once for the record
            mags = self.guides["mag"]
            t_ccds = np.full_like(mags, self.guides.t_ccd)
            if self.dyn_bgd_n_faint > 0:
                # Apply the dynamic background t_ccd bonus to the
                # dyn_bgd_n_faint faintest stars, ensuring that at least
                # MIN_DYN_BGD_ANCHOR_STARS are evaluated without the bonus. See:
                # https://nbviewer.org/urls/cxc.harvard.edu/mta/ASPECT/ipynb/misc/guide-count-dyn-bgd.ipynb
                mags = np.sort(mags)
                n_faint = min(self.dyn_bgd_n_faint, len(t_ccds))
                idx_bonus = max(len(t_ccds) - n_faint, MIN_DYN_BGD_ANCHOR_STARS)
                t_ccds[idx_bonus:] += self.dyn_bgd_dt_ccd
            self._t_ccds_bonus = t_ccds
        return self._t_ccds_bonus

    @property
    def guide_count(self):
        if not hasattr(self, "_guide_count"):
            mags = self.guides["mag"]
            self._guide_count = guide_count(mags, self.t_ccds_bonus)
        return self._guide_count

    @property
    def guide_count_9th(self):
        if not hasattr(self, "_guide_count_9th"):
            mags = self.guides["mag"]
            self._guide_count_9th = guide_count(mags, self.t_ccds_bonus, count_9th=True)
        return self._guide_count_9th

    @property
    def acq_count(self):
        if not hasattr(self, "_acq_count"):
            self._acq_count = np.sum(self.acqs["p_acq"])
        return self._acq_count

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
    def att_targ(self):
        if not hasattr(self, "_att_targ"):
            self._att_targ = self._calc_targ_from_aca(self.att, *self.target_offset)
        return self._att_targ

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

    @property
    def is_OR(self):
        """Return ``True`` if obsid corresponds to an OR."""
        if not hasattr(self, "_is_OR"):
            self._is_OR = self.obsid < 38000
        return self._is_OR

    @property
    def is_ER(self):
        """Return ``True`` if obsid corresponds to an ER."""
        return not self.is_OR

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
                    f'{star["mag"]:.2f}',
                    ha="left",
                    va="center",
                    fontsize="small",
                    color="r",
                )

    def make_starcat_plot(self):
        """Make star catalog plot for this observation."""
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

    def add_row_col(self):
        """Add row and col columns if not present"""
        if "row" in self.colnames:
            return

        row, col = yagzag_to_pixels(self["yang"], self["zang"], allow_bad=True)
        index = self.colnames.index("zang") + 1
        self.add_column(Column(row, name="row"), index=index)
        self.add_column(Column(col, name="col"), index=index + 1)

    def check_catalog(self):
        """Perform all star catalog checks."""
        for entry in self:
            entry_type = entry["type"]
            is_guide = entry_type in ("BOT", "GUI")
            is_acq = entry_type in ("BOT", "ACQ")
            is_fid = entry_type == "FID"

            if is_guide or is_fid:
                self.check_guide_fid_position_on_ccd(entry)

            if is_guide:
                star = self.guides.get_id(entry["id"])
                self.check_pos_err_guide(star)
                self.check_imposters_guide(star)
                self.check_too_bright_guide(star)
                self.check_guide_is_candidate(star)

            if is_guide or is_acq:
                self.check_bad_stars(entry)

            if is_fid:
                fid = self.fids.get_id(entry["id"])
                self.check_fid_spoiler_score(entry["idx"], fid)

        self.check_guide_overlap()
        self.check_guide_geometry()
        self.check_acq_p2()
        self.check_guide_count()
        self.check_dither()
        self.check_fid_count()
        self.check_include_exclude()

    def check_guide_overlap(self):
        """
        Check for overlapping tracked items.
        Overlap is defined as within 12 pixels.
        """
        ok = np.in1d(self["type"], ("GUI", "BOT", "FID", "MON"))
        idxs = np.flatnonzero(ok)
        for idx1, idx2 in combinations(idxs, 2):
            entry1 = self[idx1]
            entry2 = self[idx2]
            drow = entry1["row"] - entry2["row"]
            dcol = entry1["col"] - entry2["col"]
            if np.abs(drow) <= 12 and np.abs(dcol) <= 12:
                msg = (
                    "Overlapping track index (within 12 pix) "
                    f'idx [{entry1["idx"]}] and idx [{entry2["idx"]}]'
                )
                self.add_message("critical", msg)

    def check_guide_geometry(self):
        """Check for guide stars too tightly clustered.

        (1) Check for any set of n_guide-2 stars within 500" of each other.
        The nominal check here is a cluster of 3 stars within 500".  For
        ERs this check is very unlikely to fail.  For catalogs with only
        4 guide stars this will flag for any 2 nearby stars.

        This check will likely need some refinement.

        (2) Check for all stars being within 2500" of each other.

        """
        ok = np.in1d(self["type"], ("GUI", "BOT"))
        guide_idxs = np.flatnonzero(ok)
        n_guide = len(guide_idxs)

        if n_guide < 2:
            msg = f"Cannot check geometry with fewer than 2 guide stars"
            self.add_message("critical", msg)
            return

        def dist2(g1, g2):
            out = (g1["yang"] - g2["yang"]) ** 2 + (g1["zang"] - g2["zang"]) ** 2
            return out

        # First check for any set of n_guide-2 stars within 500" of each other.
        min_dist = 500
        min_dist2 = min_dist**2
        for idxs in combinations(guide_idxs, n_guide - 2):
            for idx0, idx1 in combinations(idxs, 2):
                # If any distance in this combination exceeds min_dist then
                # the combination is OK.
                if dist2(self[idx0], self[idx1]) > min_dist2:
                    break
            else:
                # Every distance was too small, issue a warning.
                cat_idxs = [idx + 1 for idx in idxs]
                msg = f'Guide indexes {cat_idxs} clustered within {min_dist}" radius'

                if self.man_angle_next > CREEP_AWAY_THRESHOLD:
                    msg += f" (man_angle_next > {CREEP_AWAY_THRESHOLD})"
                    self.add_message("critical", msg)
                else:
                    msg += f" (man_angle_next <= {CREEP_AWAY_THRESHOLD})"
                    self.add_message("warning", msg)

        # Check for all stars within 2500" of each other
        min_dist = 2500
        min_dist2 = min_dist**2
        for idx0, idx1 in combinations(guide_idxs, 2):
            if dist2(self[idx0], self[idx1]) > min_dist2:
                break
        else:
            msg = f'Guide stars all clustered within {min_dist}" radius'
            self.add_message("warning", msg)

    def check_guide_fid_position_on_ccd(self, entry):
        """Check position of guide stars and fid lights on CCD."""
        # Shortcuts and translate y/z to yaw/pitch
        dither_guide_y = self.dither_guide.y
        dither_guide_p = self.dither_guide.z

        # Set "dither" for FID to be pseudodither of 5.0 to give 1 pix margin
        # Set "track phase" dither for BOT GUI to max guide dither over
        # interval or 20.0 if undefined.  TO DO: hand the guide guide dither
        dither_track_y = 5.0 if (entry["type"] == "FID") else dither_guide_y
        dither_track_p = 5.0 if (entry["type"] == "FID") else dither_guide_p

        row_lim = ACA.max_ccd_row - ACA.CCD["window_pad"]
        col_lim = ACA.max_ccd_col - ACA.CCD["window_pad"]

        def sign(axis):
            """Return sign of the corresponding entry value.  Note that np.sign returns 0
            if the value is 0.0, not the right thing here.
            """
            return -1 if (entry[axis] < 0) else 1

        track_lims = {
            "row": (row_lim - dither_track_y * ACA.ARC_2_PIX) * sign("row"),
            "col": (col_lim - dither_track_p * ACA.ARC_2_PIX) * sign("col"),
        }

        for axis in ("row", "col"):
            track_delta = abs(track_lims[axis]) - abs(entry[axis])
            track_delta = np.round(
                track_delta, decimals=1
            )  # Official check is to 1 decimal
            for delta_lim, category in ((3.0, "critical"), (5.0, "info")):
                if track_delta < delta_lim:
                    text = (
                        f"Less than {delta_lim} pix edge margin {axis} "
                        f"lim {track_lims[axis]:.1f} "
                        f"val {entry[axis]:.1f} "
                        f"delta {track_delta:.1f}"
                    )
                    self.add_message(category, text, idx=entry["idx"])
                    break

    # TO DO: acq star position check:
    # For acq stars, the distance to the row/col padded limits are also confirmed,
    # but code to track which boundary is exceeded (row or column) is not present.
    # Note from above that the pix_row_pad used for row_lim has 7 more pixels of padding
    # than the pix_col_pad used to determine col_lim.
    # acq_edge_delta = min((row_lim - dither_acq_y / ang_per_pix) - abs(pixel_row),
    #                          (col_lim - dither_acq_p / ang_per_pix) - abs(pixel_col))
    # if ((entry_type =~ /BOT|ACQ/) and (acq_edge_delta < (-1 * 12))){
    #     push @orange_warn, sprintf "alarm [%2d] Acq Off (padded) CCD by > 60 arcsec.\n",i
    # }
    # elsif ((entry_type =~ /BOT|ACQ/) and (acq_edge_delta < 0)){
    #     push @{self->{fyi}},
    #                 sprintf "alarm [%2d] Acq Off (padded) CCD (P_ACQ should be < .5)\n",i
    # }

    def add_message(self, category, text, **kwargs):
        r"""Add message to internal messages list.

        :param category: message category ('info', 'caution', 'warning', 'critical')
        :param text: message text
        :param \**kwargs: other kwarg

        """
        message = {"text": text, "category": category}
        message.update(kwargs)
        self.messages.append(message)

    def check_acq_p2(self):
        """Check acquisition catalog safing probability."""
        P2 = -np.log10(self.acqs.calc_p_safe())
        P2 = np.round(P2, decimals=2)  # Official check is to 2 decimals
        obs_type = "OR" if self.is_OR else "ER"
        P2_lim = 2.0 if self.is_OR else 3.0
        if P2 < P2_lim:
            self.add_message(
                "critical", f"P2: {P2:.2f} less than {P2_lim} for {obs_type}"
            )
        elif P2 < P2_lim + 1:
            self.add_message(
                "warning", f"P2: {P2:.2f} less than {P2_lim + 1} for {obs_type}"
            )

    def check_include_exclude(self):
        """Check for included or excluded guide or acq stars or fids (info)"""
        call_args = self.call_args
        for typ in ("acq", "guide", "fid"):
            for action in ("include", "exclude"):
                ids = call_args.get(f"{action}_ids_{typ}")
                if ids is not None:
                    msg = f"{action}d {typ} ID(s): {ids}"

                    # Check for halfwidths.  This really only applies to
                    # include_halfws_acq, but having it here in the loop doesn't hurt.
                    halfws = call_args.get(f"{action}_halfws_{typ}")
                    if halfws is not None:
                        msg = msg + f" halfwidths(s): {halfws}"

                    self.add_message("info", msg)

    def check_guide_count(self):
        """
        Check for sufficient guide star fractional count.

        Also check for multiple very-bright stars

        """
        obs_type = "ER" if self.is_ER else "OR"
        count_9th_lim = 3.0
        if self.is_ER and np.round(self.guide_count_9th, decimals=2) < count_9th_lim:
            # Determine the threshold 9th mag equivalent value at the effective guide t_ccd
            mag9 = snr_mag_for_t_ccd(self.guides.t_ccd, 9.0, -10.9)
            self.add_message(
                "critical",
                (
                    f"{obs_type} count of 9th ({mag9:.1f} for {self.guides.t_ccd:.1f}C)"
                    f" mag guide stars {self.guide_count_9th:.2f} < {count_9th_lim}"
                ),
            )

        # Rounded guide count
        guide_count_round = np.round(self.guide_count, decimals=2)

        # Set critical guide_count threshold
        # For observations with creep-away in place as a mitigation for end of observation
        # roll error, we can accept a lower guide_count (3.5 instead of 4.0).
        # See https://occweb.cfa.harvard.edu/twiki/bin/view/Aspect/StarWorkingGroupMeeting2023x03x15
        if self.is_OR:
            count_lim = 3.5 if (self.man_angle_next <= CREEP_AWAY_THRESHOLD) else 4.0
        else:
            count_lim = 6.0

        if guide_count_round < count_lim:
            self.add_message(
                "critical",
                f"{obs_type} count of guide stars {self.guide_count:.2f} < {count_lim}",
            )
        # If in the 3.5 to 4.0 range, this probably deserves a warning.
        elif count_lim == 3.5 and guide_count_round < 4.0:
            self.add_message(
                "warning",
                f"{obs_type} count of guide stars {self.guide_count:.2f} < 4.0",
            )

        bright_cnt_lim = 1 if self.is_OR else 3
        if np.count_nonzero(self.guides["mag"] < 5.5) > bright_cnt_lim:
            self.add_message(
                "caution",
                f"{obs_type} with more than {bright_cnt_lim} stars brighter than 5.5.",
            )

        # Requested slots for guide stars and mon windows
        n_guide_or_mon_request = self.call_args["n_guide"]

        # Actual guide stars
        n_guide = len(self.guides)

        # Actual mon windows. For catalogs from pickles from proseco < 5.0
        # self.mons might be initialized to a NoneType or not be an attribute so
        # handle that as 0 monitor windows.
        try:
            n_mon = len(self.mons)
        except (TypeError, AttributeError):
            n_mon = 0

        # Different number of guide stars than requested
        if n_guide + n_mon != n_guide_or_mon_request:
            if n_mon == 0:
                # Usual case
                msg = (
                    f"{obs_type} with {n_guide} guides "
                    f"but {n_guide_or_mon_request} were requested"
                )
            else:
                msg = (
                    f"{obs_type} with {n_guide} guides and {n_mon} monitor(s) "
                    f"but {n_guide_or_mon_request} guides or mon slots were requested"
                )
            self.add_message("caution", msg)

        # Caution for any "unusual" guide star request
        typical_n_guide = 5 if self.is_OR else 8
        if n_guide_or_mon_request != typical_n_guide:
            or_mon_slots = " or mon slots" if n_mon > 0 else ""
            msg = (
                f"{obs_type} with"
                f" {n_guide_or_mon_request} guides{or_mon_slots} requested but"
                f" {typical_n_guide} is typical"
            )
            self.add_message("caution", msg)

    # Add a check that for ORs with guide count between 3.5 and 4.0, the
    # dither is 4 arcsec if dynamic background not enabled.
    def check_dither(self):
        """
        Check dither.  This presently checks that dither is 4x4 arcsec if
        dynamic background is not in use and the field has a low guide_count.
        """

        # Skip check if guide_count is 4.0 or greater
        if self.guide_count >= 4.0:
            return

        # Skip check if dynamic backround is enabled (inferred from dyn_bgd_n_faint)
        if self.dyn_bgd_n_faint > 0:
            return

        # Check that dither is <= 4x4 arcsec
        if self.dither_guide.y > 4.0 or self.dither_guide.z > 4.0:
            self.add_message(
                "critical",
                f"guide_count {self.guide_count:.2f} and dither > 4x4 arcsec",
            )

    def check_pos_err_guide(self, star):
        """Warn on stars with larger POS_ERR (warning at 1" critical at 2")"""
        agasc_id = star["id"]
        idx = self.get_id(agasc_id)["idx"]
        # POS_ERR is in milliarcsecs in the table
        pos_err = star["POS_ERR"] * 0.001
        for limit, category in ((2.0, "critical"), (1.25, "warning")):
            if np.round(pos_err, decimals=2) > limit:
                self.add_message(
                    category,
                    (
                        f"Guide star {agasc_id} POS_ERR {pos_err:.2f}, limit"
                        f" {limit} arcsec"
                    ),
                    idx=idx,
                )
                break

    def check_imposters_guide(self, star):
        """Warn on stars with larger imposter centroid offsets"""

        # Borrow the imposter offset method from starcheck
        def imposter_offset(cand_mag, imposter_mag):
            """
            For a given candidate star and the pseudomagnitude of the brightest 2x2 imposter
            calculate the max offset of the imposter counts are at the edge of the 6x6
            (as if they were in one pixel).  This is somewhat the inverse of
            proseco.get_pixmag_for_offset .
            """
            cand_counts = mag_to_count_rate(cand_mag)
            spoil_counts = mag_to_count_rate(imposter_mag)
            return spoil_counts * 3 * 5 / (spoil_counts + cand_counts)

        agasc_id = star["id"]
        idx = self.get_id(agasc_id)["idx"]
        offset = imposter_offset(star["mag"], star["imp_mag"])
        for limit, category in ((4.0, "critical"), (2.5, "warning")):
            if np.round(offset, decimals=1) > limit:
                self.add_message(
                    category,
                    f"Guide star imposter offset {offset:.1f}, limit {limit} arcsec",
                    idx=idx,
                )
                break

    def check_guide_is_candidate(self, star):
        """Critical for guide star that is not a valid guide candidate.

        This can occur for a manually included guide star.  In rare cases
        the star may still be acceptable and ACA review can accept the warning.
        """
        if not self.guides.get_candidates_mask(star):
            agasc_id = star["id"]
            idx = self.get_id(agasc_id)["idx"]
            self.add_message(
                "critical",
                f"Guide star {agasc_id} does not meet guide candidate criteria",
                idx=idx,
            )

    def check_too_bright_guide(self, star):
        """Warn on guide stars that may be too bright.

        - Critical if within 2 * mag_err of the hard 5.2 limit, caution within 3 * mag_err

        """
        agasc_id = star["id"]
        idx = self.get_id(agasc_id)["idx"]
        mag_err = star["mag_err"]
        for mult, category in ((2, "critical"), (3, "caution")):
            if star["mag"] - (mult * mag_err) < 5.2:
                self.add_message(
                    category,
                    (
                        f"Guide star {agasc_id} within {mult}*mag_err of 5.2 "
                        f"(mag_err={mag_err:.2f})"
                    ),
                    idx=idx,
                )
                break

    def check_bad_stars(self, entry):
        """Check if entry (guide or acq) is in bad star set from proseco

        :param entry: ACAReviewTable row
        :return: None
        """
        if entry["id"] in ACA.bad_star_set:
            msg = f'Star {entry["id"]} is in proseco bad star set'
            self.add_message("critical", msg, idx=entry["idx"])

    def check_fid_spoiler_score(self, idx, fid):
        """
        Check the spoiler warnings for fid

        :param idx: catalog index of fid entry being checked
        :param fid: corresponding row of ``fids`` table
        :return: None
        """
        if fid["spoiler_score"] == 0:
            return

        fid_id = fid["id"]
        category_map = {"yellow": "warning", "red": "critical"}

        for spoiler in fid["spoilers"]:
            msg = (
                f'Fid {fid_id} has {spoiler["warn"]} spoiler: star {spoiler["id"]} with'
                f' mag {spoiler["mag"]:.2f}'
            )
            self.add_message(category_map[spoiler["warn"]], msg, idx=idx)

    def check_fid_count(self):
        """
        Check for the correct number of fids.

        :return: None
        """
        obs_type = "ER" if self.is_ER else "OR"

        if len(self.fids) != self.n_fid:
            msg = (
                f"{obs_type} has {len(self.fids)} fids but {self.n_fid} were requested"
            )
            self.add_message("critical", msg)

        # Check for "typical" number of fids for an OR / ER (3 or 0)
        typical_n_fid = 3 if self.is_OR else 0
        if self.n_fid != typical_n_fid:
            msg = (
                f"{obs_type} requested {self.n_fid} fids but {typical_n_fid} is typical"
            )
            self.add_message("caution", msg)

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
        from .yoshi import get_yoshi_params_from_ocat, convert_yoshi_to_proseco_params

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
