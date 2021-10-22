# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
# ER Catalog Finder

A step forward in assistive scheduling.

## Problem statement

When building a Chandra schedule, the FOT mission planners need to choose
attitudes for ER's. The initial attitude (pitch and yaw about the
Sun line) is driven by constraints on thermal, momentum, attitude, and others.

The initial attitude they choose may not provide an acceptable star catalog
that satisfies the ACA review requirements for an ER attitude. These
requirements are somewhat more stringent than for OR's (science attitudes)
because:
- Radiation zone may have much higher ionizing radiation rates
- ER attitudes are adjustable (to a point) so we want extra margin against a
  safing action related to ACA acquisition or guide star tracking.

## Goal

Provide a flexible but fast function to find an acceptable ER
attitude and star catalog which is "near" the initial desired attitude. This
will be incorporated into the MATLAB tools OR Viewer.

## Punting half the problem back to the other team

There are two somewhat independent components to this problem:

1. Defining a set of "nearby" attitudes that are acceptable from the FOT MP
   perspective:
   1. Does not break the schedule by changing thermal or momentum "too much".
   2. Does not violate any constraint (e.g. attitude at the dwell and during
      maneuver).
2. Efficiently find an acceptable star catalog from a set of nearby attitudes.

This module provides an implementation for part (2), while assuming that the
more difficult problem of part (1) will be **solved by the FOT**.

## Plan to efficiently find an acceptable attitude from a list of attitudes

1. Get all the stars that are found in all the attitudes (one AGASC query)
2. For each attitude:
   - Select the subset of candidate guide or acq stars that land on the CCD.
   - Compute three metrics that count the effective (fractional) number of guide
     stars in complementary ways.
     - ``count_9th`` : fractional stars greater than "9th" magnitude (need 3.0)
     - ``count_10th`` : fractional stars greater than "10th" magnitude (need 6.0)
     - ``count_all`` : weighted fractional count of all stars - bigger means overall better stars
   - The first two metrics correspond directly to critical warnings in sparkles
     review (``count_9th > 3.0`` and ``count_10th > 6.0``)
   - If a star field has ``count_9th`` or ``count_10th`` too low, then toss it:
     there is no way a catalog at that attitude can pass sparkles. Conversely, if it is
     OK there is a decent chance it will end up passing the full tests.
3. Loop over remaining attitudes (with OK guide star counts) using one of two
   optimization strategies shown below. Once a catalog is found that passes
   sparkles then stop.
   1. Prioritize smallest pitch offset, binning stars in 1 deg pitch bins to minimize
     thermal impact. Within each pitch bin prioritize ``count_all``, treating any ``yaw``
     offset as equally desirable.
   2. Prioritize strictly by ``count_all``, effectively saying to start right away
     from the star fields that appear to be the best based on available stars.
   3. Prioritize by the order of attitudes passed in to the function.
"""
import numpy as np
from astropy.table import Table, MaskedColumn

from chandra_aca.transform import radec_to_yagzag, snr_mag_for_t_ccd, yagzag_to_pixels
import proseco.characteristics_guide as GUIDE
from proseco import get_aca_catalog
import agasc
from Quaternion import Quat
from chandra_aca.star_probs import guide_count
import Ska.Sun

from Ska.Sun import get_sun_pitch_yaw


def get_candidate_stars(att0, t_ccd, date=None, atts=None):
    """Get candidate stars at a given attitude.

    This gives all the stars needed to cover ``atts``.

    Parameters
    ----------
    att0 : Quaternion-like
        Central attitude
    t_ccd : float
        CCD temperature
    date : date-like
        Date of observation
    atts : list of Quaternion
        Other attitudes to try for catalog search

    Returns
    -------
    StarsTable
        Table of acceptable stars
    """
    if not isinstance(att0, Quat):
        att0 = Quat(att0)

    # Set faint mag limit based on the faintest allowed guide star at t_ccd.
    faint_mag_limit = snr_mag_for_t_ccd(t_ccd,
                                        ref_mag=GUIDE.ref_faint_mag,
                                        ref_t_ccd=GUIDE.ref_faint_mag_t_ccd)
    faint_mag_limit = min(faint_mag_limit, GUIDE.ref_faint_mag)

    # If multiple attitudes are provided then find radius on the sky that
    # contains all of them with 1.5 deg margin.
    if atts:
        dists = [agasc.sphere_dist(att0.ra, att0.dec, att.ra, att.dec) for att in atts]
        radius = np.max(dists) + 1.5
    else:
        # For a single attitude the 1.5 deg radius is fine.
        radius = 1.5

    stars = agasc.get_agasc_cone(att0.ra, att0.dec, radius=radius, date=date)

    # Copy code from proseco to filter acceptable acq and guide and stars.
    # TODO: factor these out in proseco to public functions
    acq_mask = (
        (stars['CLASS'] == 0)
        & (stars['MAG_ACA'] > 5.3)
        & (stars['MAG_ACA'] < faint_mag_limit)
        & (~np.isclose(stars['COLOR1'], 0.7))
        & (stars['MAG_ACA_ERR'] < 100)  # Mag err < 1.0 mag
        & (stars['ASPQ1'] < 40)  # Less than 2 arcsec centroid offset due to nearby spoiler
        & (stars['ASPQ2'] == 0)  # Proper motion less than 0.5 arcsec/yr
        & (stars['POS_ERR'] < 3000)  # Position error < 3.0 arcsec
        & ((stars['VAR'] == -9999) | (stars['VAR'] == 5))  # Not known to vary > 0.2 mag
    )
    guide_mask = (
        (stars['CLASS'] == 0)
        & (stars['MAG_ACA'] > 5.2)
        & (stars['MAG_ACA'] < faint_mag_limit)
        & (stars['MAG_ACA_ERR'] < 100)  # Mag err < 1.0 mag
        & (stars['ASPQ1'] < 20)  # Less than 1 arcsec offset from nearby spoiler
        & (stars['ASPQ2'] == 0)  # Proper motion less than 0.5 arcsec/yr
        & (stars['POS_ERR'] < 1250)  # Position error < 1.25 arcsec
        & ((stars['VAR'] == -9999) | (stars['VAR'] == 5))  # Not known to vary > 0.2 mag
    )
    stars['acq_mask'] = acq_mask
    stars['guide_mask'] = guide_mask

    # Only return stars that are OK for guide or acq
    ok = guide_mask | acq_mask
    stars = stars[ok]

    return stars


def filter_candidate_stars_on_ccd(att, stars):
    """Filter stars spatially to select those on the ACA CCD at ``att``

    Parameters
    ----------
    att : Quaternion-like
        Attitude
    stars : StarsTable
        Full table of candidate stars

    Returns
    -------
    StarsTable
        Stars that are on the CCD
    """
    yags, zags = radec_to_yagzag(stars['RA_PMCORR'], stars['DEC_PMCORR'], att)
    rows, cols = yagzag_to_pixels(yags, zags, allow_bad=True)
    ok = (np.abs(rows) < 507) & (np.abs(cols) < 507)  # FIXME: Hardcoded, get from characteristics?
    return stars[ok]


def get_guide_counts(mags, t_ccd):
    """
    Get guide star fractional count in various ways.

    - count_9th : fractional stars greater than "9th" magnitude (need 3.0)
    - count_10th : fractional stars greater than "10th" magnitude (need 6.0)
    - count_all : weighted fractional count of all stars

    Parameters
    ----------
    mags : np.ndarray
        Magnitudes
    t_ccd : float
        CCD temperature

    Returns
    -------
    3-tuple of floats
        count_9th, count_10th, count_all
    """
    count_9th = guide_count(mags, t_ccd, count_9th=True)
    count_10th = guide_count(mags, t_ccd, count_9th=False)

    # Generate interpolation curve for the specified input ``t_ccd``
    ref_t_ccd = -10.9

    # The 5.3 and 5.4 limits are not temperature dependent, these reflect the
    # possibility that the star will be brighter than 5.2 mag and the OBC will
    # reject it.  Note that around 6th mag mean observed catalog error is
    # around 0.1 mag.
    ref_counts = [0.0, 1.2, 1.0, 0.5, 0.0]
    ref_mags1 = [5.3, 5.4]  # Not temperature dependent
    ref_mags2 = [9.0, 10.0, 10.3]  # Temperature dependent
    ref_mags_t_ccd = (ref_mags1
                      + [snr_mag_for_t_ccd(t_ccd, ref_mag, ref_t_ccd)
                         for ref_mag in ref_mags2])

    # Do the interpolation, noting that np.interp will use the end ``counts``
    # values for any ``mag`` < ref_mags[0] or > ref_mags[-1].
    counts_t_ccd = np.interp(x=mags, xp=ref_mags_t_ccd, fp=ref_counts)
    count_all = np.sum(counts_t_ccd)

    return count_9th, count_10th, count_all


def convert_atts_to_list_of_quats(atts):
    """Convert ``atts`` to a flat list of Quat objects

    Parameters
    ----------
    atts : Quat, list
        Attitudes

    Returns
    -------
    list
        Flat list of Quat objects
    """
    if isinstance(atts, Quat):
        out = [Quat(q) for q in atts.q.reshape(-1, 4)]
    else:
        out = []
        # Assume atts is a flat list of Quats or Quat-compatible objects
        for att in atts:
            if not isinstance(att, Quat):
                att = Quat(att)
            out.append(att)
    return out


def get_att_opts_table(acar, atts):
    """Get a table of attitude options.

    This is a rich table which includes most of the details of the attitude
    finding process.

    The table is grouped by pitch bins.

    Parameters
    ----------
    acar : ACAReviewTable
        ACA catalog for candidate attitude
    atts : Quat, list
        Attitudes to consider

    Returns
    -------
    tuple of (Table, list)
        Table of attitude options and list of Quat attitudes

    """
    # Make sure atts is a flat list of Quats
    atts_list = convert_atts_to_list_of_quats(atts)
    atts_quat = Quat.from_attitude(atts_list)

    # Get the attitude and date of the initial star catalog
    att0 = acar.att
    date = acar.date

    # Get pitch and yaw of the initial attitude and the attitudes to try
    sun_ra, sun_dec = Ska.Sun.position(date)
    pitch0, yaw0 = get_sun_pitch_yaw(att0.ra, att0.dec, sun_ra=sun_ra, sun_dec=sun_dec)
    pitches, yaws = get_sun_pitch_yaw(atts_quat.ra, atts_quat.dec,
                                      sun_ra=sun_ra, sun_dec=sun_dec)

    # Delta pitch, yaw, off-nominal roll from the initial attitude. These are the absolute
    # value of the offsets, so we prioritize being closer to the initial.
    dpitches = (pitches - pitch0).ravel()
    dyaws = (yaws - yaw0).ravel()

    off_rolls = [att.roll - Ska.Sun.nominal_roll(att.ra, att.dec, sun_ra=sun_ra, sun_dec=sun_dec)
                 for att in atts_list]
    off_roll0 = att0.roll - Ska.Sun.nominal_roll(att0.ra, att0.dec, sun_ra=sun_ra, sun_dec=sun_dec)
    drolls = Quat._get_zero(off_rolls) - Quat._get_zero(off_roll0)

    # Organize the attitudes in groups by dpitch. Within each pitch group, sort
    # by dyaw.
    n_atts = len(atts_list)
    t = Table({'dpitch': dpitches,
               'dyaw': dyaws,
               'droll': drolls,
               'count_9th': np.zeros(n_atts, dtype=float),
               'count_10th': np.zeros(n_atts, dtype=float),
               'count_all': np.zeros(n_atts, dtype=float),
               'count_ok': np.zeros(n_atts, dtype=bool),
               'n_critical': MaskedColumn(np.zeros(n_atts, dtype=int), mask=True),
               'status': np.array([None] * n_atts, dtype=object),
               'att': np.array(atts_list, dtype=object),
               'acar': np.empty(n_atts, dtype=object),
               'stars': np.array([None] * n_atts, dtype=object),
               }
              )
    # Creating a numpy object array of empty lists requires this workaround
    for ii in range(n_atts):
        t['status'][ii] = list()

    for name in ['dpitch', 'dyaw', 'droll', 'count_9th', 'count_10th', 'count_all']:
        t[name].info.format = '.2f'

    # Use fanciness in Table to supply a function that formats cols dynamically
    t['att'].info.format = lambda x: f'{x.ra:.2f} {x.dec:.2f} {x.roll:.1f}'
    t['acar'].info.format = lambda x: 'acar' if x is not None else '--'
    t['stars'].info.format = lambda x: 'stars' if x is not None else '--'

    # Get all the stars covering the input attitudes
    atts_list = t['att'].tolist()
    stars = get_candidate_stars(acar.att, t_ccd=acar.t_ccd, date=acar.date, atts=atts_list)

    # Get count of 9th, 10th and mag-weighted counts
    for row in t:
        stars_att = filter_candidate_stars_on_ccd(row['att'], stars)
        mags = stars_att['MAG_ACA'][stars_att['guide_mask']]
        count_9th, count_10th, count_all = get_guide_counts(mags, acar.t_ccd)
        row['stars'] = stars_att
        row['count_9th'] = count_9th
        row['count_10th'] = count_10th
        row['count_all'] = count_all
        row['count_ok'] = row['count_9th'] >= 3.0 and row['count_10th'] >= 6.0

    return t


def get_catalog_and_review(acar, row):
    """Get an ACA catalog and review table for an attitude option.

    This updates ``row`` in-place, setting 'acar', 'n_critical', and 'status'.
    The ``row`` update propagates back to the ``att_opts`` table.

    Parameters
    ----------
    acar : ACAReviewTable
        ACA catalog for candidate attitude
    row : Table Row
        Attitude option row

    Returns
    -------

    """
    kwargs = acar.call_args.copy()
    kwargs['att'] = row['att']
    aca = get_aca_catalog(**kwargs)
    acar = aca.get_review_table()
    acar.run_aca_review()
    row['acar'] = acar
    criticals = acar.messages == 'critical'
    row['n_critical'] = len(criticals)


def find_er_catalog_by_pitch_bins(acar, att_opts, star_sets=None):
    """Find ER catalog searching within pitch bins.

    Parameters
    ----------
    acar : ACAReviewTable
        ACA catalog for candidate attitude
    att_opts : Table
        Attitude options table
    star_sets : set of tuple
        Set of star sets that have already been searched. This gets updated
        in-place if not None.

    Returns
    -------
    ACAReviewTable, None
        ACA catalog for acceptable attitude, or None if no catalog found
    """
    # Group the att_opts table in 1-deg pitch bins. Use a temp table but
    # keep track of the original table.
    tmp = att_opts.copy()
    tmp['idx'] = np.arange(len(tmp))
    tmp['pitch_abs'] = np.abs(np.round(tmp['dpitch']))

    for pitch_group in tmp.group_by('pitch_abs').groups:
        # Sort by the count metric within pitch group
        idxs = np.argsort(pitch_group['count_all'])[::-1]
        for idx in idxs:
            # Get a reference back to the correct row in atts_opts
            row = att_opts[pitch_group['idx'][idx]]
            if not row['count_ok']:
                continue
            stars_att = row['stars']

            # Keep track of stars we've seen as a sorted tuple
            if star_sets is not None:
                star_set = tuple(sorted(stars_att['AGASC_ID']))
                if star_set in star_sets:
                    continue
                star_sets.add(star_set)

            # Get the ACA catalog, do the review and set row accordingly
            get_catalog_and_review(acar, row)
            if row['n_critical'] == 0:
                row['status'].append('SEL-pitch')
                return row['acar']
    return None


def _find_er_catalog_by_idxs(acar, att_opts, star_sets=None, idxs=None):
    """Find ER catalog searching by the order in ``idxs``.

    Parameters
    ----------
    acar : ACAReviewTable
        ACA catalog for candidate attitude
    att_opts : Table
        Attitude options table
    star_sets : set of tuple, None
        Set of star sets that have already been searched. This gets updated
        in-place if not None.
    idxs : np.ndarray, None
        Index order to search (default=in input order)

    Returns
    -------
    ACAReviewTable, None
        ACA catalog for acceptable attitude, or None if no catalog found
    """
    if idxs is None:
        idxs = np.arange(len(att_opts))

    for idx in idxs:
        row = att_opts[idx]
        if not row['count_ok'] or 'criticals' in row['status']:
            continue

        # Check whether we have seen these exact stars before. If so don't bother.
        if star_sets is not None:
            stars_att = row['stars']
            stars_set = tuple(sorted(stars_att['AGASC_ID']))
            if stars_set in star_sets:
                row['status'].append('seen')
                continue
            star_sets.add(stars_set)

        # Get the ACA catalog, do the review and set row accordingly
        get_catalog_and_review(acar, row)
        if row['n_critical'] == 0:
            row['status'].append('SEL-count')
            return row['acar']

    return None


def find_er_catalog_by_count_all(acar, att_opts, star_sets=None):
    """Find ER catalog searching by the best count_all.

    Parameters
    ----------
    acar : ACAReviewTable
        ACA catalog for candidate attitude
    att_opts : Table
        Attitude options table
    star_sets : set of tuple, None
        Set of star sets that have already been searched. This gets updated
        in-place if not None.

    Returns
    -------
    ACAReviewTable, None
        ACA catalog for acceptable attitude, or None if no catalog found
    """
    idxs = np.argsort(att_opts['count_all'])[::-1]
    acar = _find_er_catalog_by_idxs(acar, att_opts, star_sets, idxs)
    return acar


def find_er_catalog_by_input_order(acar, att_opts, star_sets=None):
    """Find ER catalog searching by the best count_all.

    Parameters
    ----------
    acar : ACAReviewTable
        ACA catalog for candidate attitude
    att_opts : Table
        Attitude options table
    star_sets : set of tuple, None
        Set of star sets that have already been searched. This gets updated
        in-place if not None.

    Returns
    -------
    ACAReviewTable, None
        ACA catalog for acceptable attitude, or None if no catalog found
    """
    acar = _find_er_catalog_by_idxs(acar, att_opts, star_sets)
    return acar


# Registry of catalog finders by algorithm name
FIND_ER_CATALOG_FUNCS = {'pitch_bins': find_er_catalog_by_pitch_bins,
                         'count_all': find_er_catalog_by_count_all,
                         'input_order': find_er_catalog_by_input_order}


def find_er_catalog(acar, atts, alg='input_order', check_star_sets=True):
    """Find an ER catalog for a list of nearby attitude options.

    Parameters
    ----------
    acar : ACAReviewTable
        ACA catalog for initial attitude
    atts : list, Quat
        Attitudes to search for for ER catalog
    alg : str, optional
        Algorithm to use to find the ER catalog ('input_order', 'pitch_bins', 'count_all')
    check_star_sets : bool, optional
        Check if star set at an attitude has been seen before and skip (i.e. change
        in attitude does not change the candidate guide/acq stars in the FOV)
        (default=True).

    Returns
    -------
    tuple of (ACAReviewTable, attitude options Table)
        ACA review table for the ER catalog (or None if not found) and attitude
        options table.
    """
    att_opts = get_att_opts_table(acar, atts)

    # Set of star catalogs that we have already tried. If a small change in
    # attitude does not bring in new stars then it is unlikely to succeed if
    # the previous failed.
    star_sets = set() if check_star_sets else None

    # Initialize the output to None. This gets set if we find an ER catalog
    find_func = FIND_ER_CATALOG_FUNCS[alg]
    acar_out = find_func(acar, att_opts, star_sets)

    return acar_out, att_opts
