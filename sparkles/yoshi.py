import numpy as np
import numpy.ma
from chandra_aca.transform import calc_aca_from_targ
from chandra_aca.drift import get_aca_offsets
from mica.archive.cda import get_ocat_web, get_ocat_local
from Ska.Sun import nominal_roll
from cxotime import CxoTime


def get_yoshi_params_from_ocat(obsid, obs_date=None, web_ocat=True):
    """
    For an obsid in the OCAT, fetch params from OCAT and define a few defaults
    for the standard info needed to get an ACA attitude and run
    yoshi / proseco / sparkles.

    :param obsid: obsid
    :param obs_date: intended date. If None, use the date from the OCAT if possible
        else use current date.
    :param web_ocat: use the web version of the OCAT (uses get_ocat_local if False)
    :returns: dictionary of target parameters/keywords from OCAT.  Can be used with
              convert_yoshi_to_proseco_params .
    """

    if web_ocat:
        ocat = get_ocat_web(obsid=obsid)
    else:
        ocat = get_ocat_local(obsid=obsid)

    if obs_date is None and ocat["start_date"] is not numpy.ma.masked:
        # If obsid has a defined start_date then use that. Otherwise if obsid is
        # not assigned an LTS bin then start_date will be masked and we fall
        # through to CxoTime(None) which is NOW.
        obs_date = CxoTime(ocat["start_date"]).date
    else:
        obs_date = CxoTime(obs_date).date

    targ = {
        "obs_date": obs_date,
        "detector": ocat["instr"],
        "ra_targ": ocat["ra"],
        "dec_targ": ocat["dec"],
        "offset_y": ocat["y_off"],
        "offset_z": ocat["z_off"],
    }

    # Leaving focus offset as not-implemented
    targ["focus_offset"] = 0

    # For sim offsets, leave default at 0 unless explicit in OCAT
    # OCAT entry is in mm
    targ["sim_offset"] = 0
    if ocat["z_sim"] is not numpy.ma.masked:
        targ["sim_offset"] = ocat["z_sim"] * 397.7225924607

    # Could use get_target_aimpoint but that needs icxc if not on HEAD
    # and we don't need to care that much for future observations.
    aimpoints = {
        "ACIS-I": (970.0, 975.0, 3),
        "ACIS-S": (210.0, 520.0, 7),
        "HRC-S": (2195.0, 8915.0, 2),
        "HRC-I": (7590.0, 7745.0, 0),
    }
    chip_x, chip_y, chip_id = aimpoints[ocat["instr"]]
    targ.update({"chipx": chip_x, "chipy": chip_y, "chip_id": chip_id})

    # Nominal roll is not quite the same targ and aca but don't care.
    targ["roll_targ"] = nominal_roll(ocat["ra"], ocat["dec"], obs_date)

    # Set dither from defaults and override if defined in OCAT
    if ocat["instr"].startswith("ACIS"):
        targ.update({"dither_y": 8, "dither_z": 8})
    else:
        targ.update({"dither_y": 20, "dither_z": 20})

    if ocat["dither"] == "Y":
        targ.update(
            {"dither_y": ocat["y_amp"] * 3600, "dither_z": ocat["z_amp"] * 3600}
        )
    if ocat["dither"] == "N":
        targ.update({"dither_y": 0, "dither_z": 0})

    return targ


def run_one_yoshi(
    *,
    obsid,
    detector,
    chipx,
    chipy,
    chip_id,
    ra_targ,
    dec_targ,
    roll_targ,
    offset_y,
    offset_z,
    sim_offset,
    focus_offset,
    dither_y,
    dither_z,
    obs_date,
    t_ccd,
    man_angle,
    **kwargs
):
    """
    Run proseco and sparkles for an observation request in a roll/temperature/man_angle
    scenario.
    :param obsid: obsid
    :param detector: detector (ACIS-I|ACIS-S|HRC-I|HRC-S)
    :param chipx: chipx from zero-offset aimpoint table entry for obsid
    :param chipy: chipy from zero-offset aimpoint table entry for obsid
    :param chip_id: chip_id from zero-offset aimpoint table entry for obsid
    :param ra_targ: target RA (degrees)
    :param dec_targ: target Dec (degrees)
    :param roll_targ: target Roll (degrees)
    :param offset_y: target offset_y (arcmin)
    :param offset_z: target offset_z (arcmin)
    :param sim_offset: SIM Z offset (steps)
    :param focus_offset: SIM focus offset (steps)
    :param dither_y: Y amplitude dither (arcsec)
    :param dither_z: Z amplitude dither (arcsec)
    :param obs_date: observation date (for proper motion and ACA offset projection)
    :param t_ccd: ACA CCD temperature (degrees C)
    :param man_angle: maneuver angle (degrees)
    :param **kwargs: additional keyword args to update or override params from
        yoshi for call to get_aca_catalog()
    :returns: dictionary of (ra_aca, dec_aca, roll_aca,
                             n_critical, n_warning, n_caution, n_info,
                             P2, guide_count)
    """
    from proseco import get_aca_catalog

    params = convert_yoshi_to_proseco_params(
        obsid,
        detector,
        chipx,
        chipy,
        chip_id,
        ra_targ,
        dec_targ,
        roll_targ,
        offset_y,
        offset_z,
        sim_offset,
        focus_offset,
        dither_y,
        dither_z,
        obs_date,
        t_ccd,
        man_angle,
        **kwargs,
    )

    aca = get_aca_catalog(**params)
    acar = aca.get_review_table()
    acar.run_aca_review()
    q_aca = aca.att

    # Get values for report
    report = {
        "ra_aca": q_aca.ra,
        "dec_aca": q_aca.dec,
        "roll_aca": q_aca.roll,
        "n_critical": len(acar.messages == "critical"),
        "n_warning": len(acar.messages == "warning"),
        "n_caution": len(acar.messages == "caution"),
        "n_info": len(acar.messages == "info"),
        "P2": -np.log10(acar.acqs.calc_p_safe()),
        "guide_count": acar.guide_count,
    }

    return report


def convert_yoshi_to_proseco_params(
    obsid,
    detector,
    chipx,
    chipy,
    chip_id,
    ra_targ,
    dec_targ,
    roll_targ,
    offset_y,
    offset_z,
    sim_offset,
    focus_offset,
    dither_y,
    dither_z,
    obs_date,
    t_ccd,
    man_angle,
    **kwargs,
):
    """
    Convert yoshi parameters to equivalent proseco arguments

    :param obsid: obsid (used only for labeling)
    :param detector: detector (ACIS-I|ACIS-S|HRC-I|HRC-S)
    :param chipx: chipx from zero-offset aimpoint table entry for obsid
    :param chipy: chipy from zero-offset aimpoint table entry for obsid
    :param chip_id: chip_id from zero-offset aimpoint table entry for obsid
    :param ra_targ: target RA (degrees)
    :param dec_targ: target Dec (degrees)
    :param roll_targ: target Roll (degrees)
    :param offset_y: target offset_y (arcmin)
    :param offset_z: target offset_z (arcmin)
    :param sim_offset: SIM Z offset (steps)
    :param focus_offset: SIM focus offset (steps)
    :param dither_y: Y amplitude dither (arcsec)
    :param dither_z: Z amplitude dither (arcsec)
    :param obs_date: observation date (for proper motion and ACA offset projection)
    :param t_ccd: ACA CCD temperature (degrees C)
    :param man_angle: maneuver angle (degrees)
    :param **kwargs: extra keyword arguments which update the output proseco params
    :returns: dictionary of keyword arguments for proseco

    """
    if offset_y is np.ma.masked:
        offset_y = 0.0
    if offset_z is np.ma.masked:
        offset_z = 0.0
    if sim_offset is np.ma.masked:
        sim_offset = 0.0
    if focus_offset is np.ma.masked:
        focus_offset = 0.0

    # Calculate dynamic offsets using the supplied temperature.
    aca_offset_y, aca_offset_z = get_aca_offsets(
        detector, chip_id, chipx, chipy, obs_date, t_ccd
    )

    # Get the ACA quaternion using target offsets and dynamic offsets.
    # Note that calc_aca_from_targ expects target offsets in degrees and obs is now in arcmin
    q_aca = calc_aca_from_targ(
        (ra_targ, dec_targ, roll_targ),
        (offset_y / 60.0) + (aca_offset_y / 3600.0),
        (offset_z / 60.0) + (aca_offset_z / 3600.0),
    )

    # Get keywords for proseco
    out = dict(
        obsid=obsid,
        att=q_aca,
        man_angle=man_angle,
        date=obs_date,
        t_ccd=t_ccd,
        dither=(dither_y, dither_z),
        detector=detector,
        sim_offset=sim_offset,
        focus_offset=focus_offset,
        n_acq=8,
        n_guide=5,
        n_fid=3,
    )

    out.update(kwargs)

    return out
