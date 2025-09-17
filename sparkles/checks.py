# Licensed under a 3-clause BSD style license - see LICENSE.rst
import functools
from itertools import combinations

import numpy as np
import proseco.characteristics as ACA
from chandra_aca.transform import mag_to_count_rate, snr_mag_for_t_ccd
from proseco.core import ACACatalogTableRow, StarsTableRow

from sparkles.aca_check_table import ACACheckTable
from sparkles.messages import Message

# Observations with man_angle_next less than or equal to CREEP_AWAY_THRESHOLD
# are considered "creep away" observations. CREEP_AWAY_THRESHOLD is in units of degrees.
CREEP_AWAY_THRESHOLD = 3.0


def acar_check_wrapper(func):
    """Wrapper to call check functions with ACAReviewTable.

    Checks in checks are written to return a list of messages, while the checks
    in sparkles.checks are written to add messages to the ACAReviewTable. This wrapper
    converts the former to the latter.
    """

    @functools.wraps(func)
    def wrapper(acar: ACACheckTable, *args, **kwargs):
        msgs: list[Message] = func(acar, *args, **kwargs)
        messages = [
            {
                key: val
                for key in ("category", "text", "idx")
                if (val := getattr(msg, key)) is not None
            }
            for msg in msgs
        ]

        acar.messages.extend(messages)

    return wrapper


def check_guide_overlap(acar: ACACheckTable) -> list[Message]:
    """Check for overlapping tracked items.

    Overlap is defined as within 12 pixels.
    """
    msgs = []
    ok = np.in1d(acar["type"], ("GUI", "BOT", "FID", "MON"))
    idxs = np.flatnonzero(ok)
    for idx1, idx2 in combinations(idxs, 2):
        entry1 = acar[idx1]
        entry2 = acar[idx2]
        drow = entry1["row"] - entry2["row"]
        dcol = entry1["col"] - entry2["col"]
        if np.abs(drow) <= 12 and np.abs(dcol) <= 12:
            msg = (
                "Overlapping track index (within 12 pix) "
                f"idx [{entry1['idx']}] and idx [{entry2['idx']}]"
            )
            msgs += [Message("critical", msg)]
    return msgs


def check_run_jupiter_checks(acar: ACACheckTable) -> list[Message]:
    """
    Run jupiter checks.

    This is a wrapper to run the jupiter checks.  It first checks if the date
    is excluded from jupiter checks.  If not, it gets the jupiter position
    and then runs the individual jupiter checks.

    Parameters
    ----------
    acar : ACACheckTable
        The ACA review table to check.

    Returns
    -------
    list of Message
        List of messages from the jupiter checks.
    """
    msgs = []
    if "jupiter" not in acar.target_name.lower():
        return msgs

    from proseco import jupiter

    # First check for exclude dates when Jupiter is fainter than the Optically Bright
    # Object limit of -2.0 mag.
    if jupiter.date_is_excluded(acar.date):
        return msgs

    msgs += check_jupiter_on_ccd(acar)
    msgs += check_jupiter_acq_spoilers(acar)
    msgs += check_jupiter_track_spoilers(acar)
    msgs += check_jupiter_distribution(acar)

    msgs += [Message("info", "Ran Jupiter checks")]
    return msgs


def check_jupiter_on_ccd(acar: ACACheckTable) -> list[Message]:
    """
    Check if Jupiter is on the CCD.

    Parameters
    ----------
    acar : ACACheckTable
        The ACA review table to check.

    Returns
    -------
    list of Message
        List of messages from the jupiter on CCD check.
    """
    msgs = []
    if len(acar.jupiter) == 0:
        msgs += [
            Message(
                "warning",
                f"Jupiter not on CCD, expected for target '{acar.target_name}'",
            )
        ]
    return msgs


def check_jupiter_acq_spoilers(acar: ACACheckTable) -> list[Message]:
    """
    Check for columns spoiled by Jupiter in acquisition boxes.

    This uses a 15 column pad around Jupiter.

    It does not explicitly use an estimate of maneuver error.

    Parameters
    ----------
    acar : ACACheckTable
        The ACA review table to check.

    Returns
    -------
    list of Message
        List of messages from the jupiter acquisition box check.
    """
    from proseco.jupiter import get_jupiter_acq_pos

    msgs = []
    ok = np.in1d(acar["type"], ("BOT", "ACQ"))
    acqs = acar[ok]
    pad = 15

    _, jcol = get_jupiter_acq_pos(acar.date, acar.jupiter)
    if jcol is None:
        return []

    # For each acquisition box confirm a column spoiled by jupiter isn't in it
    for entry in acqs:
        col_min = entry["col"] - entry["halfw"] / 5
        col_max = entry["col"] + entry["halfw"] / 5
        in_box = (jcol + pad >= col_min) & (jcol - pad <= col_max)
        if np.any(in_box):
            msg = (
                f"Jupiter column in acquisition box idx {entry['idx']} id {entry['id']}"
                f" row {entry['row']:.1f} col {entry['col']:.1f}"
            )
            msgs += [Message("critical", msg, idx=entry["idx"])]
    return msgs


def check_jupiter_track_spoilers(acar: ACACheckTable) -> list[Message]:
    """
    Check for Jupiter spoiling stars or fids.

    This uses proseco.jupiter.check_spoiled_by_jupiter to determine
    if any tracked stars or fids are spoiled by Jupiter.

    Parameters
    ----------
    acar : ACACheckTable
        The ACA review table to check.

    Returns
    -------
    list of Message
        List of messages from the jupiter tracked star check.
    """
    from proseco.jupiter import check_spoiled_by_jupiter

    msgs = []
    ok = np.in1d(acar["type"], ("GUI", "BOT", "FID"))
    guide_and_fid = acar[ok]
    spoiled, _ = check_spoiled_by_jupiter(guide_and_fid, acar.jupiter)
    for row in guide_and_fid[spoiled]:
        msg = f"Jupiter spoils tracked star idx {row['idx']} id {row['id']}"
        msgs += [Message("critical", msg, idx=row["idx"])]
    return msgs


def check_jupiter_distribution(acar: ACACheckTable) -> list[Message]:
    """
    Check for guide star distribution for Jupiter fields.

    The guideline requires at least 2 guide stars on the CCD half opposite
    Jupiter, one side of the CCD is positive in row and the other negative.
    If the padded Jupiter crosses the row=0 line during the observation
    then at least 2 guide stars are required on each side.

    This uses proseco.jupiter.jupiter_distribution_check.

    Parameters
    ----------
    acar : ACACheckTable
        The ACA review table to check.

    Returns
    -------
    list of Message
        List of messages from the jupiter guide star distribution check.
    """
    from proseco.jupiter import jupiter_distribution_check

    # Check that there are at least 2 guide stars in each quadrant of the ccd
    msgs = []
    ok = np.in1d(acar["type"], ("GUI", "BOT"))
    if not jupiter_distribution_check(acar[ok], acar.jupiter):
        msg = (
            "Jupiter guide star distribution check failed. "
            "Need 2 guide stars always opposite Jupiter."
        )
        msgs += [Message("critical", msg)]
    return msgs


def check_guide_geometry(acar: ACACheckTable) -> list[Message]:
    """Check for guide stars too tightly clustered.

    (1) Check for any set of n_guide-2 stars within 500" of each other.
    The nominal check here is a cluster of 3 stars within 500".  For
    ERs this check is very unlikely to fail.  For catalogs with only
    4 guide stars this will flag for any 2 nearby stars.

    This check will likely need some refinement.

    (2) Check for all stars being within 2500" of each other.

    """
    msgs = []
    ok = np.in1d(acar["type"], ("GUI", "BOT"))
    guide_idxs = np.flatnonzero(ok)
    n_guide = len(guide_idxs)

    if n_guide < 2:
        msg = "Cannot check geometry with fewer than 2 guide stars"
        msgs += [Message("critical", msg)]
        return msgs

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
            if dist2(acar[idx0], acar[idx1]) > min_dist2:
                break
        else:
            # Every distance was too small, issue a warning.
            cat_idxs = [idx + 1 for idx in idxs]
            msg = f'Guide indexes {cat_idxs} clustered within {min_dist}" radius'

            if acar.man_angle_next > CREEP_AWAY_THRESHOLD:
                msg += f" (man_angle_next > {CREEP_AWAY_THRESHOLD})"
                msgs += [Message("critical", msg)]
            else:
                msg += f" (man_angle_next <= {CREEP_AWAY_THRESHOLD})"
                msgs += [Message("warning", msg)]

    # Check for all stars within 2500" of each other
    min_dist = 2500
    min_dist2 = min_dist**2
    for idx0, idx1 in combinations(guide_idxs, 2):
        if dist2(acar[idx0], acar[idx1]) > min_dist2:
            break
    else:
        msg = f'Guide stars all clustered within {min_dist}" radius'
        msgs += [Message("warning", msg)]
    return msgs


def check_guide_fid_position_on_ccd(
    acar: ACACheckTable, entry: ACACatalogTableRow
) -> list[Message]:
    """Check position of guide stars and fid lights on CCD."""
    msgs = []
    # Shortcuts and translate y/z to yaw/pitch
    dither_guide_y = acar.dither_guide.y
    dither_guide_p = acar.dither_guide.z

    # Set "dither" for FID to be pseudodither of 5.0 to give 1 pix margin
    # Set "track phase" dither for BOT GUI to max guide dither over
    # interval or 20.0 if undefined.  TO DO: hand the guide guide dither
    dither_track_y = 5.0 if (entry["type"] == "FID") else dither_guide_y
    dither_track_p = 5.0 if (entry["type"] == "FID") else dither_guide_p

    row_lim = ACA.max_ccd_row - ACA.CCD["window_pad"]
    col_lim = ACA.max_ccd_col - ACA.CCD["window_pad"]

    def sign(axis):
        """Return sign of the corresponding entry value.

        Note that np.sign returns 0 if the value is 0.0, not the right thing here.
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
                msgs += [Message(category, text, idx=entry["idx"])]
                break
    return msgs


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
#     push @{acar->{fyi}},
#                 sprintf "alarm [%2d] Acq Off (padded) CCD (P_ACQ should be < .5)\n",i
# }


def check_acq_p2(acar: ACACheckTable) -> list[Message]:
    """Check acquisition catalog safing probability."""
    msgs = []
    P2 = -np.log10(acar.acqs.calc_p_safe())
    P2 = np.round(P2, decimals=2)  # Official check is to 2 decimals
    obs_type = "OR" if acar.is_OR else "ER"
    P2_lim = 2.0 if acar.is_OR else 3.0
    if P2 < P2_lim:
        msgs += [Message("critical", f"P2: {P2:.2f} less than {P2_lim} for {obs_type}")]
    elif P2 < P2_lim + 1:
        msgs += [
            Message("warning", f"P2: {P2:.2f} less than {P2_lim + 1} for {obs_type}")
        ]
    return msgs


def check_include_exclude(acar: ACACheckTable) -> list[Message]:
    """Check for included or excluded guide or acq stars or fids (info)"""
    msgs = []
    call_args = acar.call_args
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

                msgs += [Message("info", msg)]
    return msgs


def check_guide_count(acar: ACACheckTable) -> list[Message]:
    """
    Check for sufficient guide star fractional count.

    Also check for multiple very-bright stars

    """
    msgs = []
    obs_type = "ER" if acar.is_ER else "OR"
    count_9th_lim = 3.0
    if acar.is_ER and np.round(acar.guide_count_9th, decimals=2) < count_9th_lim:
        # Determine the threshold 9th mag equivalent value at the effective guide t_ccd
        mag9 = snr_mag_for_t_ccd(acar.guides.t_ccd, 9.0, -10.9)
        msgs += [
            Message(
                "critical",
                (
                    f"{obs_type} count of 9th ({mag9:.1f} for {acar.guides.t_ccd:.1f}C)"
                    f" mag guide stars {acar.guide_count_9th:.2f} < {count_9th_lim}"
                ),
            )
        ]

    # Rounded guide count
    guide_count_round = np.round(acar.guide_count, decimals=2)

    # Set critical guide_count threshold
    # For observations with creep-away in place as a mitigation for end of observation
    # roll error, we can accept a lower guide_count (3.5 instead of 4.0).
    # See https://occweb.cfa.harvard.edu/twiki/bin/view/Aspect/StarWorkingGroupMeeting2023x03x15
    if acar.is_OR:
        count_lim = 3.5 if (acar.man_angle_next <= CREEP_AWAY_THRESHOLD) else 4.0
    else:
        count_lim = 6.0

    if guide_count_round < count_lim:
        msgs += [
            Message(
                "critical",
                f"{obs_type} count of guide stars {acar.guide_count:.2f} < {count_lim}",
            )
        ]
    # If in the 3.5 to 4.0 range, this probably deserves a warning.
    elif count_lim == 3.5 and guide_count_round < 4.0:
        msgs += [
            Message(
                "warning",
                f"{obs_type} count of guide stars {acar.guide_count:.2f} < 4.0",
            )
        ]

    bright_cnt_lim = 1 if acar.is_OR else 3
    if np.count_nonzero(acar.guides["mag"] < 5.5) > bright_cnt_lim:
        msgs += [
            Message(
                "caution",
                f"{obs_type} with more than {bright_cnt_lim} stars brighter than 5.5.",
            )
        ]

    # Requested slots for guide stars and mon windows
    n_guide_or_mon_request = acar.call_args["n_guide"]

    # Actual guide stars
    n_guide = len(acar.guides)

    # Actual mon windows. For catalogs from pickles from proseco < 5.0
    # acar.mons might be initialized to a NoneType or not be an attribute so
    # handle that as 0 monitor windows.
    try:
        n_mon = len(acar.mons)
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
        msgs += [Message("caution", msg)]

    # Caution for any "unusual" guide star request
    typical_n_guide = 5 if acar.is_OR else 8
    if n_guide_or_mon_request != typical_n_guide:
        or_mon_slots = " or mon slots" if n_mon > 0 else ""
        msg = (
            f"{obs_type} with"
            f" {n_guide_or_mon_request} guides{or_mon_slots} requested but"
            f" {typical_n_guide} is typical"
        )
        msgs += [Message("caution", msg)]

    return msgs


# Add a check that for ORs with guide count between 3.5 and 4.0, the
# dither is 4 arcsec if dynamic background not enabled.
def check_dither(acar: ACACheckTable) -> list[Message]:
    """Check dither.

    This presently checks that dither is 4x4 arcsec if dynamic background is not in
    use and the field has a low guide_count.
    """
    msgs = []
    # Skip check if guide_count is 4.0 or greater
    if acar.guide_count >= 4.0:
        return msgs

    # Skip check if dynamic backround is enabled (inferred from dyn_bgd_n_faint)
    if acar.dyn_bgd_n_faint > 0:
        return msgs

    # Check that dither is <= 4x4 arcsec
    if acar.dither_guide.y > 4.0 or acar.dither_guide.z > 4.0:
        msgs += [
            Message(
                "critical",
                f"guide_count {acar.guide_count:.2f} and dither > 4x4 arcsec",
            )
        ]

    return msgs


def check_config_for_no_guide_dither(acar: ACACheckTable) -> list[Message]:
    """Check special configurations for no guide dither."""
    msgs = []
    if np.round(acar.dither_guide.y, 0) == 0 and np.round(acar.dither_guide.z, 0) == 0:
        if acar.dyn_bgd_n_faint > 0:
            msgs += [
                Message(
                    "critical",
                    "guide_dither close to 0 arcsec and dyn_bgd_n_faint > 0",
                )
            ]
        if acar.man_angle_next > CREEP_AWAY_THRESHOLD:
            msgs += [
                Message(
                    "critical",
                    f"guide_dither close to 0 arcsec and man_angle_next > {CREEP_AWAY_THRESHOLD}",
                )
            ]
    return msgs


def check_pos_err_guide(acar: ACACheckTable, star: StarsTableRow) -> list[Message]:
    """Warn on stars with larger POS_ERR (warning at 1" critical at 2")"""
    msgs = []
    agasc_id = star["id"]
    idx = acar.get_id(agasc_id)["idx"]
    # POS_ERR is in milliarcsecs in the table
    pos_err = star["POS_ERR"] * 0.001
    for limit, category in ((2.0, "critical"), (1.25, "warning")):
        if np.round(pos_err, decimals=2) > limit:
            msgs += [
                Message(
                    category,
                    (
                        f"Guide star {agasc_id} POS_ERR {pos_err:.2f}, limit"
                        f" {limit} arcsec"
                    ),
                    idx=idx,
                )
            ]
            break
    return msgs


def check_imposters_guide(acar: ACACheckTable, star: StarsTableRow) -> list[Message]:
    """Warn on stars with larger imposter centroid offsets"""

    # Borrow the imposter offset method from starcheck
    def imposter_offset(cand_mag, imposter_mag):
        """Get imposter offset.

        For a given candidate star and the pseudomagnitude of the brightest 2x2
        imposter calculate the max offset of the imposter counts are at the edge of
        the 6x6 (as if they were in one pixel).  This is somewhat the inverse of
        proseco.get_pixmag_for_offset.
        """
        cand_counts = mag_to_count_rate(cand_mag)
        spoil_counts = mag_to_count_rate(imposter_mag)
        return spoil_counts * 3 * 5 / (spoil_counts + cand_counts)

    msgs = []
    agasc_id = star["id"]
    idx = acar.get_id(agasc_id)["idx"]
    offset = imposter_offset(star["mag"], star["imp_mag"])
    for limit, category in ((4.0, "critical"), (2.5, "warning")):
        if np.round(offset, decimals=1) > limit:
            msgs += [
                Message(
                    category,
                    f"Guide star imposter offset {offset:.1f}, limit {limit} arcsec",
                    idx=idx,
                )
            ]
            break
    return msgs


def check_guide_is_candidate(acar: ACACheckTable, star: StarsTableRow) -> list[Message]:
    """Critical for guide star that is not a valid guide candidate.

    This can occur for a manually included guide star.  In rare cases
    the star may still be acceptable and ACA review can accept the warning.
    """
    msgs = []
    if not acar.guides.get_candidates_mask(star):
        agasc_id = star["id"]
        idx = acar.get_id(agasc_id)["idx"]
        msgs += [
            Message(
                "critical",
                f"Guide star {agasc_id} does not meet guide candidate criteria",
                idx=idx,
            )
        ]
    return msgs


def check_too_bright_guide(acar: ACACheckTable, star: StarsTableRow) -> list[Message]:
    """Warn on guide stars that may be too bright.

    - Critical if within 2 * mag_err of the hard 5.2 limit, caution within 3 * mag_err

    """
    msgs = []
    agasc_id = star["id"]
    idx = acar.get_id(agasc_id)["idx"]
    mag_err = star["mag_err"]
    for mult, category in ((2, "critical"), (3, "caution")):
        if star["mag"] - (mult * mag_err) < 5.2:
            msgs += [
                Message(
                    category,
                    (
                        f"Guide star {agasc_id} within {mult}*mag_err of 5.2 "
                        f"(mag_err={mag_err:.2f})"
                    ),
                    idx=idx,
                )
            ]
            break
    return msgs


def check_bad_stars(entry: ACACatalogTableRow) -> list[Message]:
    """Check if entry (guide or acq) is in bad star set from proseco

    :param entry: ACATable row
    :return: None
    """
    msgs = []
    if entry["id"] in ACA.bad_star_set:
        msg = f"Star {entry['id']} is in proseco bad star set"
        msgs += [Message("critical", msg, idx=entry["idx"])]
    return msgs


def check_fid_spoiler_score(idx, fid) -> list[Message]:
    """
    Check the spoiler warnings for fid

    :param idx: catalog index of fid entry being checked
    :param fid: corresponding row of ``fids`` table
    :return: None
    """
    msgs = []
    if fid["spoiler_score"] == 0:
        return msgs

    fid_id = fid["id"]
    category_map = {"yellow": "warning", "red": "critical"}

    for spoiler in fid["spoilers"]:
        msg = (
            f"Fid {fid_id} has {spoiler['warn']} spoiler: star {spoiler['id']} with"
            f" mag {spoiler['mag']:.2f}"
        )
        msgs += [Message(category_map[spoiler["warn"]], msg, idx=idx)]
    return msgs


def check_fid_count(acar: ACACheckTable) -> list[Message]:
    """
    Check for the correct number of fids.

    :return: None
    """
    msgs = []
    obs_type = "ER" if acar.is_ER else "OR"

    if len(acar.fids) != acar.n_fid:
        msg = f"{obs_type} has {len(acar.fids)} fids but {acar.n_fid} were requested"
        msgs += [Message("critical", msg)]

    # Check for "typical" number of fids for an OR / ER (3 or 0)
    typical_n_fid = 3 if acar.is_OR else 0
    if acar.n_fid != typical_n_fid:
        msg = f"{obs_type} requested {acar.n_fid} fids but {typical_n_fid} is typical"
        msgs += [Message("caution", msg)]

    return msgs
