import gzip
import pickle
from pathlib import Path

import numpy as np
import proseco.characteristics as ACA
import pytest
import ska_sun
from proseco import get_aca_catalog
from proseco.characteristics import MonCoord, MonFunc
from proseco.core import StarsTable
from proseco.tests.test_common import DARK40, mod_std_info
from Quaternion import Quat
from ska_sun import apply_sun_pitch_yaw, nominal_roll

from sparkles import ACAReviewTable, run_aca_review

KWARGS_48464 = {
    "att": [-0.51759295, -0.30129397, 0.27093045, 0.75360213],
    "date": "2019:031:13:25:30.000",
    "detector": "ACIS-S",
    "dither_acq": (7.9992, 7.9992),
    "dither_guide": (7.9992, 7.9992),
    "man_angle": 67.859,
    "n_acq": 8,
    "n_fid": 0,
    "n_guide": 8,
    "obsid": 48464,
    "sim_offset": -3520.0,
    "focus_offset": 0,
    "t_ccd_acq": -9.943,
    "t_ccd_guide": -9.938,
    "dyn_bgd_n_faint": 0,
}


def test_jupiter_present():
    """Test that Jupiter checks run nominally for an observation in 2025 with Jupiter
    present and in target name"""
    kwargs = {
        "att": [0.47935711, 0.44950136, 0.68474659, 0.31509901],
        "date": "2025:258:14:42:16.000",
        "detector": "HRC-S",
        "dither_acq": (20, 20),
        "dither_guide": (20, 20),
        "man_angle": 180,
        "n_acq": 8,
        "n_fid": 3,
        "n_guide": 5,
        "obsid": 29674,
        "sim_offset": 0,
        "focus_offset": 0,
        "t_ccd_acq": -8.6,
        "t_ccd_guide": -8.6,
        "dyn_bgd_n_faint": 2,
        # These last two kwargs are required for Jupiter checks
        "target_name": "ALL ABOUT JUPITER",
        "duration": 10000,
    }
    aca = get_aca_catalog(**kwargs)
    acar = aca.get_review_table()
    acar.run_aca_review(make_html=False)
    assert acar.messages[-1] == {"category": "info", "text": "Ran Jupiter checks"}


def test_jupiter_not_present():
    kwargs = KWARGS_48464.copy()
    # This target name "accidentally has jupiter in it"
    kwargs["target_name"] = "NO JUPITER HERE"
    kwargs["duration"] = 30000
    kwargs["t_ccd_acq"] = -10
    kwargs["t_ccd_guide"] = -10
    aca = get_aca_catalog(**kwargs)
    acar = aca.get_review_table()
    acar.run_aca_review(make_html=False)
    # Confirm one of the messages is {'category': 'critical', 'text': 'No Jupiter position found'}
    assert any(
        msg == {"category": "critical", "text": "No Jupiter position found"}
        for msg in acar.messages
    )


def test_t_ccd_effective_message():
    """Test printing a message about effective guide and/or acq CCD temperature
    when it is different from the predicted temperature."""
    kwargs = KWARGS_48464.copy()
    kwargs["t_ccd_guide"] = ACA.aca_t_ccd_penalty_limit + 0.75
    kwargs["t_ccd_acq"] = ACA.aca_t_ccd_penalty_limit + 0.5
    aca = get_aca_catalog(**kwargs)
    acar = aca.get_review_table()
    acar.run_aca_review()

    # Pre-formatted text that gets put into HTML report
    text = acar.get_text_pre()

    eff_guide = (
        kwargs["t_ccd_guide"]
        + 1
        + (kwargs["t_ccd_guide"] - ACA.aca_t_ccd_penalty_limit)
    )
    eff_acq = (
        kwargs["t_ccd_acq"] + 1 + (kwargs["t_ccd_acq"] - ACA.aca_t_ccd_penalty_limit)
    )
    assert (
        f"Predicted Guide CCD temperature (max): {kwargs['t_ccd_guide']:.1f} "
        f'<span class="caution">(Effective : {eff_guide:.1f})</span>' in text
    )
    assert (
        f"Predicted Acq CCD temperature (init) : {kwargs['t_ccd_acq']:.1f} "
        f'<span class="caution">(Effective : {eff_acq:.1f})</span>' in text
    )


def test_review_catalog(proseco_agasc_1p7, tmpdir):
    aca = get_aca_catalog(**KWARGS_48464)
    acar = aca.get_review_table()
    acar.run_aca_review()
    assert acar.messages == [
        {
            "category": "warning",
            "text": "Guide star imposter offset 2.6, limit 2.5 arcsec",
            "idx": 4,
        },
        {"category": "warning", "text": "P2: 3.33 less than 4.0 for ER"},
        {
            "category": "critical",
            "text": "ER count of 9th (8.9 for -9.9C) mag guide stars 1.91 < 3.0",
        },
        {"category": "critical", "text": "ER count of guide stars 5.00 < 6.0"},
        {"category": "caution", "text": "ER with 5 guides but 8 were requested"},
    ]

    assert acar.roll_options is None

    msgs = acar.messages >= "critical"
    assert msgs == [
        {
            "category": "critical",
            "text": "ER count of 9th (8.9 for -9.9C) mag guide stars 1.91 < 3.0",
        },
        {"category": "critical", "text": "ER count of guide stars 5.00 < 6.0"},
    ]

    assert acar.review_status() == -1

    # Run the review but without making the HTML and ensure review messages
    # are available on roll options.
    acar.run_aca_review(roll_level="critical", roll_args={"method": "uniq_ids"})
    assert len(acar.roll_options) > 1
    assert acar.roll_options[0]["acar"].messages == acar.messages
    assert len(acar.roll_options[1]["acar"].messages) > 0

    # Check doing a full review for this obsid
    acar = aca.get_review_table()
    acar.run_aca_review(
        make_html=True,
        report_dir=tmpdir,
        report_level="critical",
        roll_level="critical",
        roll_args={"method": "uniq_ids"},
    )

    path = Path(str(tmpdir))
    assert (path / "index.html").exists()
    obspath = path / "obs48464"
    assert (obspath / "acq" / "index.html").exists()
    assert (obspath / "guide" / "index.html").exists()
    assert (obspath / "rolls" / "index.html").exists()


def test_review_roll_options():
    """
    Test that the 'acar' key in the roll_option dict is an ACAReviewTable
    and that the first one has the same messages as the base (original roll)
    version

    :param tmpdir: temp dir supplied by pytest
    :return: None
    """
    # This is a catalog that has a critical message and one roll option
    kwargs = {
        "att": (160.9272490316051, 14.851572261604668, 99.996111473617802),
        "date": "2019:046:07:16:58.449",
        "detector": "ACIS-S",
        "dither_acq": (7.9992, 7.9992),
        "dither_guide": (7.9992, 7.9992),
        "focus_offset": 0.0,
        "man_angle": 1.792525648258372,
        "n_acq": 8,
        "n_fid": 3,
        "n_guide": 5,
        "obsid": 21477,
        "sim_offset": 0.0,
        "t_ccd_acq": -11.14616454993262,
        "t_ccd_guide": -11.150381856818923,
    }

    aca = get_aca_catalog(**kwargs)
    acar = aca.get_review_table()
    acar.run_aca_review(roll_level="critical")

    assert len(acar.roll_options) == 4

    # First roll_option is at the same attitude (and roll) as original.  The check
    # code is run again independently but the outcome should be the same.
    assert acar.roll_options[0]["acar"].messages == acar.messages

    for opt in acar.roll_options:
        assert isinstance(opt["acar"], ACAReviewTable)


def test_probs_weak_reference():
    """
    Test issues related to the weak reference to self.acqs within the AcqProbs
    objects in cand_acqs.

    See comment in ACAReviewTable.__init__() for details.

    """
    aca = get_aca_catalog(**KWARGS_48464)

    aca2 = pickle.loads(pickle.dumps(aca))
    assert aca2.acqs is not aca.acqs

    # These fail.  TODO: fix!
    # aca2 = aca.__class__(aca)  # default is copy=True
    # aca2 = deepcopy(aca)

    acar = ACAReviewTable(aca)

    assert aca.guides is not acar.guides
    assert aca.acqs is not acar.acqs


def test_roll_options_with_include_ids():
    """
    Test case from James that was breaking code due to a roll option that puts
    a force_include star outside the FOV.

    """
    kwargs = {
        "obsid": 48397.0,
        "att": [0.43437703, -0.47822201, -0.68470554, 0.33734053],
        "date": "2019:053:04:05:33.004",
        "detector": "ACIS-S",
        "dither_acq": (7.9992, 2.0016),
        "dither_guide": (7.9992, 2.0016),
        "man_angle": 131.2011858838081,
        "n_acq": 8,
        "n_fid": 0,
        "n_guide": 8,
        "sim_offset": 0.0,
        "focus_offset": 0.0,
        "t_ccd_acq": -12.157792574498563,
        "t_ccd_guide": -12.17,
        "include_ids_acq": np.array(  # Also tests passing float ids for include
            [
                8.13042280e08,
                8.13040960e08,
                8.13044168e08,
                8.12911064e08,
                8.12920176e08,
                8.12913936e08,
                8.13043216e08,
                8.13045352e08,
            ]
        ),
        "include_halfws_acq": np.array(
            [160.0, 160.0, 160.0, 160.0, 160.0, 160.0, 120.0, 60.0]
        ),
    }

    aca = get_aca_catalog(**kwargs)
    acar = aca.get_review_table()
    acar.run_aca_review(roll_level="all", roll_args={"method": "uniq_ids"})
    # As of the 2020-02 acq model update there is just one roll option
    # assert len(acar.roll_options) > 1


def test_roll_options_with_monitor_star():
    """
    Test that roll optimization succeeds in the case where a monitor star is
    specified in the catalog. This tests https://github.com/sot/proseco/issues/365
    which fixes https://github.com/sot/proseco/issues/364.
    """
    kwargs = {
        "att": [0.32916333, -0.50759709, 0.07481427, 0.79271655],
        "date": "2021:116:16:42:09.065",
        "detector": "ACIS-I",
        "dither_acq": (7.9992, 7.9992),
        "dither_guide": (7.9992, 7.9992),
        "man_angle": 86.49934496445843,
        "n_acq": 8,
        "n_fid": 3,
        "n_guide": 5,
        "obsid": 23050.0,
        "sim_offset": 0.0,
        "focus_offset": 0.0,
        "t_ccd_acq": -9.067548914167258,
        "t_ccd_guide": -8.616156814261098,
        "t_ccd_penalty_limit": -6.8,
        "monitors": [[335.516667, 58.675556, MonCoord.RADEC, 9.63, MonFunc.AUTO]],
    }
    aca = get_aca_catalog(**kwargs)
    acar = aca.get_review_table()
    acar.run_aca_review()
    assert acar.messages == []

    # get_roll_options throws BadMonitorError before proseco PR#365
    acar.get_roll_options()


def test_roll_options_use_dyn_bgd_n_faint():
    """Test that the roll options use dyn_bgd_n_faint in the catalog
    selection and the n_stars value.  This test uses an artificially large
    dyn_bgd_t_ccd of -10 accentuate the differences."""

    kwargs = {
        "att": [-0.63787389, 0.07940607, -0.10873119, 0.75828036],
        "date": "2024:316",
        "detector": "ACIS-S",
        "dither": (16, 16),
        "focus_offset": 0,
        "sim_offset": 0,
        "man_angle": 5,
        "n_acq": 8,
        "n_fid": 3,
        "n_guide": 5,
        "obsid": 30580,
        "t_ccd": -6.0,
    }

    aca1_n_faint = 0
    aca1 = get_aca_catalog(**kwargs, dyn_bgd_n_faint=aca1_n_faint, dyn_bgd_dt_ccd=-10)
    acar1 = aca1.get_review_table()
    acar1.get_roll_options()

    aca2_n_faint = 3
    aca2 = get_aca_catalog(**kwargs, dyn_bgd_n_faint=aca2_n_faint, dyn_bgd_dt_ccd=-10)
    acar2 = aca2.get_review_table()
    acar2.get_roll_options()

    assert acar1.dyn_bgd_n_faint == aca1_n_faint
    assert acar2.dyn_bgd_n_faint == aca2_n_faint
    for roll_option in acar1.roll_options:
        assert roll_option["acar"].dyn_bgd_n_faint == aca1_n_faint
    for roll_option in acar2.roll_options:
        assert roll_option["acar"].dyn_bgd_n_faint == aca2_n_faint

    aca1_n_stars = [roll_option["n_stars"] for roll_option in acar1.roll_options]
    aca2_n_stars = [roll_option["n_stars"] for roll_option in acar2.roll_options]

    # In these cases the roll options are the same.
    assert len(aca1_n_stars) == len(aca2_n_stars)

    # But the guide count of the last one is not.
    assert np.allclose(
        np.array(aca1_n_stars),
        np.array([1.5302331230204111, 1.5302331230204111, 2.5306029287208744]),
    )
    assert np.allclose(
        np.array(aca2_n_stars),
        np.array([1.5302331230204111, 1.5302331230204111, 3.530615379948673]),
    )


def test_uniform_roll_options(proseco_agasc_1p7):
    """Use obsid 22508 as a test case for failing to find a roll option using
    the 'uniq_ids' algorithm and falling through to a 'uniform' search.

    See https://github.com/sot/sparkles/issues/138 for context.
    """
    kwargs = {
        "att": [-0.25019352, -0.90540872, -0.21768747, 0.26504794],
        "date": "2020:045:18:19:50.234",
        "detector": "ACIS-S",
        "n_guide": 5,
        "n_fid": 3,
        "dither": 8.0,
        "focus_offset": 0,
        "man_angle": 1.56,
        "obsid": 22508,
        "sim_offset": 0,
        "t_ccd_acq": -9.8,
        "t_ccd_guide": -9.8,
        "dyn_bgd_n_faint": 0,
    }

    aca = get_aca_catalog(**kwargs)
    acar = aca.get_review_table()
    acar.run_aca_review(
        roll_level="critical", roll_args={"max_roll_dev": 2.5, "d_roll": 0.25}
    )

    # Fell through to uniform roll search
    assert acar.roll_info["method"] == "uniform"

    # Found at least one roll option with no critical messages
    assert any(
        len(roll_option["acar"].messages >= "critical") == 0
        for roll_option in acar.roll_options
    )

    assert len(acar.roll_options) == 5

    # Now limit the number of roll options
    acar = aca.get_review_table()
    acar.run_aca_review(
        roll_level="critical",
        roll_args={"max_roll_dev": 2.5, "max_roll_options": 3, "d_roll": 0.25},
    )
    assert len(acar.roll_options) == 3


def test_catch_exception_from_function():
    exc = run_aca_review(raise_exc=False, load_name="non-existent load name fail fail")
    assert "FileNotFoundError: no matching pickle file" in exc

    with pytest.raises(FileNotFoundError):
        exc = run_aca_review(load_name="non-existent load name fail fail")


def test_catch_exception_from_method():
    aca = get_aca_catalog(**KWARGS_48464)
    acar = aca.get_review_table()
    exc = acar.run_aca_review(raise_exc=False, roll_level="BAD VALUE")
    assert "ValueError: tuple.index(x): x not in tuple" in exc

    with pytest.raises(ValueError):
        acar.run_aca_review(roll_level="BAD VALUE")


def test_run_aca_review_function(proseco_agasc_1p7, tmpdir):
    aca = get_aca_catalog(**KWARGS_48464)
    acar = aca.get_review_table()
    acars = [acar]

    exc = run_aca_review(load_name="test_load", report_dir=tmpdir, acars=acars)

    assert exc is None
    assert acar.messages == [
        {
            "category": "warning",
            "text": "Guide star imposter offset 2.6, limit 2.5 arcsec",
            "idx": 4,
        },
        {"category": "warning", "text": "P2: 3.33 less than 4.0 for ER"},
        {
            "category": "critical",
            "text": "ER count of 9th (8.9 for -9.9C) mag guide stars 1.91 < 3.0",
        },
        {"category": "critical", "text": "ER count of guide stars 5.00 < 6.0"},
        {"category": "caution", "text": "ER with 5 guides but 8 were requested"},
    ]

    path = Path(str(tmpdir))
    assert (path / "index.html").exists()
    obspath = path / "obs48464"
    assert (obspath / "starcat48464.png").exists()
    assert "TEST_LOAD sparkles review" in (path / "index.html").read_text()


def test_run_aca_review_dyn_bgd_n_faint(proseco_agasc_1p7, tmpdir):
    aca = get_aca_catalog(**KWARGS_48464)
    acar = aca.get_review_table()
    acars = [acar]

    exc = run_aca_review(
        load_name="test_load",
        report_dir=tmpdir,
        acars=acars,
        dyn_bgd_n_faint=2,
    )

    assert exc is None
    assert acar.dyn_bgd_n_faint == 2
    assert np.isclose(acar.guide_count, 5, atol=0.1)

    # Assert same warnings as test_run_aca_review_function but with a different
    # guide count and new info message
    assert acar.messages == [
        {"text": "Using dyn_bgd_n_faint=2 (call_args val=0)", "category": "info"},
        {
            "category": "warning",
            "text": "Guide star imposter offset 2.6, limit 2.5 arcsec",
            "idx": 4,
        },
        {"category": "warning", "text": "P2: 3.33 less than 4.0 for ER"},
        {
            "category": "critical",
            "text": "ER count of 9th (8.9 for -9.9C) mag guide stars 1.91 < 3.0",
        },
        {"category": "critical", "text": "ER count of guide stars 5.00 < 6.0"},
        {"category": "caution", "text": "ER with 5 guides but 8 were requested"},
    ]


def test_roll_outside_range():
    """
    Run a test on obsid 48334 from ~MAR1119 that is at a pitch that has 0 roll_dev
    and is at a roll that is not exactly nominal roll for the attitude and date.
    The 'att' ends up with roll outside of the internally-computed roll_min / roll_max
    which caused indexing excption in the roll-options code.  Fixed by PR #91 which
    includes code  to expand the roll_min / roll_max range to always include the roll
    of the originally supplied attitude.
    """
    kw = {
        "att": [-0.82389459, -0.1248412, 0.35722113, 0.42190692],
        "date": "2019:073:21:55:30.000",
        "detector": "ACIS-S",
        "dither_acq": (7.9992, 7.9992),
        "dither_guide": (7.9992, 7.9992),
        "focus_offset": 0.0,
        "man_angle": 122.97035882921071,
        "n_acq": 8,
        "n_fid": 0,
        "n_guide": 8,
        "obsid": 48334.0,
        "sim_offset": 0.0,
        "t_ccd_acq": -10.257559323423214,
        "t_ccd_guide": -10.25810835536192,
    }
    aca = get_aca_catalog(**kw)
    acar = aca.get_review_table()
    acar.get_roll_options()
    assert Quat(kw["att"]).roll <= acar.roll_info["roll_max"]
    assert Quat(kw["att"]).roll >= acar.roll_info["roll_min"]


def test_roll_options_dec89_9(proseco_agasc_1p7):
    """Test getting roll options for an OR and ER at very high declination
    where the difference between the ACA and target frames is large.  Here
    the roll will differ by around 10 deg.
    """
    dec = 89.9
    date = "2019:006:12:00:00"
    roll = ska_sun.nominal_roll(0, dec, time=date)
    att = Quat([0, dec, roll])

    # Expected roll options.  Note same basic outputs for add_ids and drop_ids but
    # difference roll values.

    # NOTE ALSO: the P2 values are impacted by the bad region aroundrow=-318,
    # col=-298. If handling of that bad region for acq changes then the P2
    # values may change.
    exp = {}
    exp[48000] = [
        " roll   P2  n_stars improvement roll_min roll_max  add_ids  drop_ids",
        "------ ---- ------- ----------- -------- -------- --------- --------",
        "286.97 3.61    0.55        0.00   286.97   286.97        --       --",
        "280.97 7.23    7.67       10.35   276.72   285.22 608567744       --",
        "287.47 7.25    6.27        8.69   276.14   297.81        --       --",
    ]

    exp[18000] = [
        " roll   P2  n_stars improvement roll_min roll_max  add_ids  drop_ids",
        "------ ---- ------- ----------- -------- -------- --------- --------",
        "276.71 3.61    7.85        0.00   276.71   276.71        --       --",
        "277.09 7.25    8.00        1.84   265.88   287.54        --       --",
        "270.59 7.15    8.00        1.80   266.21   274.96 608567744       --",
    ]

    for obsid in (48000, 18000):
        kwargs = mod_std_info(att=att, n_guide=8, n_fid=0, obsid=obsid, date=date)
        # Exclude a bunch of good stars to make the initial catalog lousy
        exclude_ids = [606470536, 606601760, 606732552, 606732584, 610926712, 611058024]
        kwargs["exclude_ids_acq"] = exclude_ids
        kwargs["exclude_ids_guide"] = exclude_ids

        aca = get_aca_catalog(**kwargs)
        acar = aca.get_review_table()
        acar.run_aca_review(
            roll_level="all", roll_args={"method": "uniq_ids"}, make_html=False
        )
        tbl = acar.get_roll_options_table()
        out = tbl.pformat(max_lines=-1, max_width=-1)
        assert out == exp[obsid]


def test_calc_targ_from_aca():
    """
    Confirm _calc_targ_from_aca seems to do the right thing based on obsid
    This does a bit too much processing for what should be a lightweight test.
    """
    # Testing an ER where we expect the targ quaternion to be the same as the ACA
    # quaternion.
    aca = get_aca_catalog(**KWARGS_48464)
    acar = aca.get_review_table()
    q_targ = acar._calc_targ_from_aca(acar.att, 0, 0)
    assert q_targ is acar.att

    # Here we change the review object to represent an OR (by changing is_OR) and
    # confirm the targ quaternion is different from the ACA quaternion
    acar._is_OR = True
    q_targ = acar._calc_targ_from_aca(acar.att, 0, 0)
    # Specifically the targ quaternion should be off by ODB_SI_ALIGN which is about 70
    # arcsecs in yaw
    assert np.isclose(acar.att.dq(q_targ).yaw * 3600, 69.59, atol=0.01, rtol=0)


def test_get_roll_intervals(proseco_agasc_1p7):
    """
    Test that the behavior of get_roll_intervals is different for ORs and ERs with
    regard to use of offsets.  They are set to arbitrary large values in the test.
    """

    # This uses the catalog at KWARGS_48464, but would really be better as a fully
    # synthetic test
    obs_kwargs = KWARGS_48464.copy()
    # Use these values to override the get_roll_intervals ranges to get more interesting
    # outputs.
    obs_kwargs["target_offset"] = (20 / 60.0, 30 / 60)  # deg
    aca_er = get_aca_catalog(**obs_kwargs)
    acar_er = aca_er.get_review_table()

    kw_or = obs_kwargs.copy()
    # Set this one to have an OR obsid (and not 0 which is special)
    kw_or["obsid"] = 1
    aca_or = get_aca_catalog(**kw_or)
    acar_or = aca_or.get_review_table()

    max_roll_dev = 5

    with pytest.warns(FutureWarning):
        er_roll_intervs, er_info = acar_er.get_roll_intervals(
            acar_er.get_candidate_better_stars(), roll_dev=max_roll_dev
        )

    er_roll_intervs, er_info = acar_er.get_roll_intervals(
        acar_er.get_candidate_better_stars(), max_roll_dev=max_roll_dev
    )

    or_roll_intervs, or_info = acar_or.get_roll_intervals(
        acar_or.get_candidate_better_stars(), max_roll_dev=max_roll_dev
    )

    assert acar_er.att.roll <= er_info["roll_max"]
    assert acar_er.att.roll >= er_info["roll_min"]

    # The roll ranges in ACA rolls should be different for the ER and the OR version
    assert or_info != er_info

    # Up to this point this is really a weak functional test.  The following asserts
    # are more regression tests for the attitude at obsid 48464
    or_rolls = [interv["roll"] for interv in or_roll_intervs]
    er_rolls = [interv["roll"] for interv in er_roll_intervs]
    assert or_rolls != er_rolls

    # Set a function to do some looping and isclose logic to compare
    # the actual vs expected intervals.
    def compare_intervs(intervs, exp_intervs):
        assert len(intervs) == len(exp_intervs)
        for interv, exp_interv in zip(intervs, exp_intervs):
            assert interv.keys() == exp_interv.keys()
            for key in interv:
                if key.startswith("roll"):
                    assert np.isclose(interv[key], exp_interv[key], atol=1e-6, rtol=0)
                else:
                    assert interv[key] == exp_interv[key]

    # For the OR we expect this
    or_exp_intervs = [
        {
            "roll": 281.56864473302164,
            "roll_min": 281.56864473302164,
            "roll_max": 281.5759766065589,
            "add_ids": {84943288},
            "drop_ids": {84937736},
        },
        {
            "roll": 289.0759766065589,
            "roll_min": 283.8259766065589,
            "roll_max": 291.56864473302164,
            "add_ids": set(),
            "drop_ids": set(),
        },
        {
            "roll": 289.0759766065589,
            "roll_min": 287.8259766065589,
            "roll_max": 290.3259766065589,
            "add_ids": {84941648},
            "drop_ids": set(),
        },
        {
            "roll": 289.8259766065589,
            "roll_min": 289.3259766065589,
            "roll_max": 290.3259766065589,
            "add_ids": {85328120, 84941648},
            "drop_ids": set(),
        },
        {
            "roll": 291.56864473302164,
            "roll_min": 289.3259766065589,
            "roll_max": 291.56864473302164,
            "add_ids": {85328120},
            "drop_ids": set(),
        },
    ]

    compare_intervs(or_roll_intervs, or_exp_intervs)

    # For the ER we expect these
    er_exp_intervs = [
        {
            "roll": 290.8033828990559,
            "roll_min": 285.1783828990559,
            "roll_max": 291.67104636183814,
            "add_ids": set(),
            "drop_ids": set(),
        },
        {
            "roll": 291.67104636183814,
            "roll_min": 289.6783828990559,
            "roll_max": 291.67104636183814,
            "add_ids": {84943288},
            "drop_ids": set(),
        },
        {
            "roll": 291.67104636183814,
            "roll_min": 290.9283828990559,
            "roll_max": 291.67104636183814,
            "add_ids": {85328120, 84943288},
            "drop_ids": set(),
        },
    ]
    compare_intervs(er_roll_intervs, er_exp_intervs)


def test_roll_options_for_not_allowed_pitch():
    """
    Confirm that roll option determination works without error
    for a pitch that is not allowed (no options returned).
    """
    ra = dec = 0.0
    date = "2023:001"
    roll = nominal_roll(ra, dec, date)
    att0 = Quat([ra, dec, roll])

    att = apply_sun_pitch_yaw(att0, pitch=40, time=date)
    aca = get_aca_catalog(**mod_std_info(att=att, date=date))
    acar = aca.get_review_table()
    acar.get_roll_options()
    assert len(acar.roll_options) == 1
    tbl = acar.get_roll_options_table()
    assert len(tbl) == 1


def test_review_with_mon_star():
    """Test that requesting n_guide=5 with a monitor window produces no review
    messages (all OK), although this does result in aca.n_guide == 4."""

    monitors = [[0, 0, MonCoord.YAGZAG, 7.5, MonFunc.MON_FIXED]]

    stars = StarsTable.empty()
    stars.add_fake_constellation(n_stars=8, mag=8.5)
    aca = get_aca_catalog(
        **mod_std_info(n_fid=3, n_guide=5, obsid=5000),
        monitors=monitors,
        stars=stars,
        dark=DARK40,
        raise_exc=True,
    )
    acar = ACAReviewTable(aca)
    acar.run_aca_review()

    assert aca.n_guide == 4
    assert len(aca.mons) == 1
    assert acar.messages == []


def test_review_from_pickle():
    """
    Test that run_aca_review works from a gzip pickle file.
    """

    # This is similar to the loading from core.get_acas_from_pickle
    # but that method does not allow full path specification.
    filename = Path(__file__).parent / "data" / "proseco_4.12.1_catalog.pkl.gz"
    with gzip.open(filename, "rb") as fh:
        acas_dict = pickle.load(fh)
    acars = [ACAReviewTable(aca, obsid=obsid) for obsid, aca in acas_dict.items()]
    acars[0].run_aca_review()
    assert acars[0].messages == []
