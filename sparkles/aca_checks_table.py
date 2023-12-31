# Licensed under a 3-clause BSD style license - see LICENSE.rst
from itertools import chain

import numpy as np
from astropy.table import Column
from chandra_aca.star_probs import guide_count
from chandra_aca.transform import yagzag_to_pixels
from proseco.catalog import ACATable
from proseco.core import MetaAttribute

from sparkles.messages import MessagesList

# Minimum number of "anchor stars" that are always evaluated *without* the bonus
# from dynamic background when dyn_bgd_n_faint > 0. This is mostly to avoid the
# situation where 4 stars are selected and 2 are faint bonus stars. In this case
# there would be only 2 anchor stars that ensure good tracking even without
# dyn bgd.
MIN_DYN_BGD_ANCHOR_STARS = 3


def get_t_ccds_bonus(mags, t_ccd, dyn_bgd_n_faint, dyn_bgd_dt_ccd):
    """Return array of t_ccds with dynamic background bonus applied.

    This adds ``dyn_bgd_dt_ccd`` to the effective CCD temperature for the
    ``dyn_bgd_n_faint`` faintest stars, ensuring that at least MIN_DYN_BGD_ANCHOR_STARS
    are evaluated without the bonus. See:
    https://nbviewer.org/urls/cxc.harvard.edu/mta/ASPECT/ipynb/misc/guide-count-dyn-bgd.ipynb

    :param mags: array of star magnitudes
    :param t_ccd: single T_ccd value (degC)
    :param dyn_bgd_n_faint: number of faintest stars to apply bonus
    :returns: array of t_ccds (matching ``mags``) with dynamic background bonus applied
    """
    t_ccds = np.full_like(mags, t_ccd)

    # If no bonus stars then just return the input t_ccd broadcast to all stars
    if dyn_bgd_n_faint == 0:
        return t_ccds

    idxs = np.argsort(mags)
    n_faint = min(dyn_bgd_n_faint, len(t_ccds))
    idx_bonus = max(len(t_ccds) - n_faint, MIN_DYN_BGD_ANCHOR_STARS)
    for idx in idxs[idx_bonus:]:
        t_ccds[idx] += dyn_bgd_dt_ccd

    return t_ccds


class ACAChecksTable(ACATable):
    messages = MetaAttribute()

    def __init__(self, *args, **kwargs):
        obsid = kwargs.pop("obsid", None)

        super().__init__(*args, **kwargs)

        self.messages = MessagesList()

        # If no data were provided then skip all the rest of the initialization.
        # This happens during slicing. The result is not actually
        # a functional ACAReviewTable, but it allows inspection of data.
        if len(self.colnames) == 0:
            return

        # Add row and col columns from yag/zag, if not already there.
        self.add_row_col()

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
            self._t_ccds_bonus = get_t_ccds_bonus(
                self.guides["mag"],
                self.guides.t_ccd,
                self.dyn_bgd_n_faint,
                self.dyn_bgd_dt_ccd,
            )
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

    @property
    def att_targ(self):
        if not hasattr(self, "_att_targ"):
            self._att_targ = self._calc_targ_from_aca(self.att, *self.target_offset)
        return self._att_targ

    def add_row_col(self):
        """Add row and col columns if not present"""
        if "row" in self.colnames:
            return

        row, col = yagzag_to_pixels(self["yang"], self["zang"], allow_bad=True)
        index = self.colnames.index("zang") + 1
        self.add_column(Column(row, name="row"), index=index)
        self.add_column(Column(col, name="col"), index=index + 1)
