# coding: utf-8
# Licensed under a 3-clause BSD style license - see LICENSE.rst
import copy
import pickle

import agasc
import numpy as np
import pytest
from chandra_aca.transform import mag_to_count_rate
from proseco import get_aca_catalog
from proseco.characteristics import CCD, MonCoord, MonFunc
from proseco.core import StarsTable
from proseco.tests.test_common import DARK40, STD_INFO, mod_std_info
from Quaternion import Quat

from sparkles import ACAReviewTable, get_t_ccds_bonus


def test_check_slice_index():
    """Test slice and index"""
    stars = StarsTable.empty()
    stars.add_fake_constellation(n_stars=8, mag=10.25)
    aca = get_aca_catalog(**STD_INFO, stars=stars, dark=DARK40)
    acar = aca.get_review_table()
    for item in (1, slice(5, 10)):
        acar1 = acar[item]
        assert acar1.colnames == acar.colnames
        for name in acar1.colnames:
            assert np.all(acar1[name] == acar[name][item])


def test_check_P2():
    """Test the check of acq P2"""
    stars = StarsTable.empty()
    stars.add_fake_constellation(n_stars=3, mag=10.25)
    aca = get_aca_catalog(**STD_INFO, stars=stars, dark=DARK40)
    acar = ACAReviewTable(aca)

    # Check P2 for an OR (default obsid=0)
    acar.check_acq_p2()
    assert len(acar.messages) == 1
    msg = acar.messages[0]
    assert msg["category"] == "critical"
    assert "less than 2.0 for OR" in msg["text"]

    # Check P2 constructed for an ER with stars intended to have P2 > 2 and < 3
    stars = StarsTable.empty()
    stars.add_fake_constellation(n_stars=5, mag=9.5)
    aca = get_aca_catalog(
        **mod_std_info(n_fid=0, n_guide=8, obsid=50000),
        stars=stars,
        dark=DARK40,
        raise_exc=True
    )
    acar = ACAReviewTable(aca)
    acar.check_acq_p2()
    assert len(acar.messages) == 1
    msg = acar.messages[0]
    assert msg["category"] == "critical"
    assert "less than 3.0 for ER" in msg["text"]


def test_n_guide_check_not_enough_stars():
    """Test the check that number of guide stars selected is as requested"""

    stars = StarsTable.empty()
    stars.add_fake_constellation(n_stars=4, mag=8.5)
    aca = get_aca_catalog(
        **mod_std_info(n_fid=3, n_guide=5, obsid=5000),
        stars=stars,
        dark=DARK40,
        raise_exc=True
    )
    acar = ACAReviewTable(aca)
    acar.check_guide_count()
    assert acar.messages == [
        {"text": "OR with 4 guides but 5 were requested", "category": "caution"}
    ]


def test_guide_is_candidate():
    """Test the check that guide star meets candidate star requirements

    Make a star catalog with a CLASS=3 star and force include it for guide.
    """

    stars = StarsTable.empty()
    stars.add_fake_constellation(n_stars=6, mag=8.5, CLASS=[3, 0, 0, 0, 0, 0])
    aca = get_aca_catalog(
        **mod_std_info(n_fid=3, n_guide=5, obsid=5000),
        stars=stars,
        dark=DARK40,
        include_ids_guide=[100],
        raise_exc=True
    )
    acar = ACAReviewTable(aca)
    acar.check_catalog()
    assert acar.messages == [
        {
            "text": "Guide star 100 does not meet guide candidate criteria",
            "category": "critical",
            "idx": 8,
        },
        {"text": "included guide ID(s): [100]", "category": "info"},
    ]


def test_n_guide_check_atypical_request():
    """Test the check that number of guide stars selected is typical"""

    stars = StarsTable.empty()
    stars.add_fake_constellation(n_stars=8, mag=8.5)
    aca = get_aca_catalog(
        **mod_std_info(n_fid=3, n_guide=4, obsid=5000),
        stars=stars,
        dark=DARK40,
        raise_exc=True
    )
    acar = ACAReviewTable(aca)
    acar.check_guide_count()
    assert acar.messages == [
        {"text": "OR with 4 guides requested but 5 is typical", "category": "caution"}
    ]


def test_n_guide_mon_check_atypical_request():
    """Test the check that number of guide stars selected is typical
    in the case where are monitors"""

    stars = StarsTable.empty()
    stars.add_fake_constellation(n_stars=8, mag=8.5)
    monitors = [[50, -50, MonCoord.YAGZAG, 7.5, MonFunc.MON_TRACK]]

    aca = get_aca_catalog(
        **mod_std_info(n_fid=2, n_guide=6, obsid=5000, monitors=monitors),
        stars=stars,
        dark=DARK40,
        raise_exc=True
    )
    acar = ACAReviewTable(aca)
    acar.check_guide_count()
    assert acar.messages == [
        {
            "text": "OR with 6 guides or mon slots requested but 5 is typical",
            "category": "caution",
        }
    ]


vals = [
    (5, 4.07, 4.66),  # n_guide, legacy_guide_count, dyn_bdg_guide_count
    (4, 3.25, 3.60),  # 4 stars, legacy and dyn bdg guide counts closer together
    (3, 2.42, 2.42),  # 3 stars, legacy and dyn bdg equal (no bonus stars)
]


@pytest.mark.parametrize("vals", vals)
def test_guide_count_dyn_bgd_bonus(vals):
    n_guide, leg_guide_count, dyn_guide_count = vals
    stars = StarsTable.empty()

    stars.add_fake_constellation(
        mag=np.linspace(10, 10.2, n_guide), size=2000, n_stars=n_guide
    )

    aca_leg = get_aca_catalog(**STD_INFO, dark=DARK40, stars=stars, dyn_bgd_n_faint=0)
    aca_dyn = get_aca_catalog(
        **STD_INFO, dark=DARK40, stars=stars, dyn_bgd_n_faint=2, dyn_bgd_dt_ccd=-4.0
    )
    # Same catalog but with different attributes
    assert len(aca_leg.guides) == n_guide
    assert len(aca_dyn.guides) == n_guide
    assert np.all(aca_leg.guides["mag"] == aca_dyn.guides["mag"])

    acar_leg = ACAReviewTable(aca_leg)
    acar_dyn = ACAReviewTable(aca_dyn)
    # Computed guide counts without / with dyn_bgd_n_faint=2
    assert np.isclose(acar_leg.guide_count, leg_guide_count, rtol=0, atol=0.1)
    assert np.isclose(acar_dyn.guide_count, dyn_guide_count, rtol=0, atol=0.1)


def test_n_guide_too_few_guide_or_mon():
    """Test the check that the number of actual guide and mon stars is what
    was requested"""

    stars = StarsTable.empty()
    stars.add_fake_constellation(n_stars=4, mag=8.5)
    monitors = [[50, -50, MonCoord.YAGZAG, 7.5, MonFunc.MON_TRACK]]

    aca = get_aca_catalog(
        **mod_std_info(n_fid=2, n_guide=6, obsid=5000, monitors=monitors),
        stars=stars,
        dark=DARK40,
        raise_exc=True
    )
    acar = ACAReviewTable(aca)
    acar.check_guide_count()
    assert acar.messages == [
        {
            "category": "caution",
            "text": (
                "OR with 4 guides and 1 monitor(s) but 6 guides or mon slots were"
                " requested"
            ),
        },
        {
            "category": "caution",
            "text": "OR with 6 guides or mon slots requested but 5 is typical",
        },
    ]


def test_guide_count_er1():
    """Test the check that an ER has enough fractional guide stars by guide_count"""

    # This configuration should have not enough bright stars
    stars = StarsTable.empty()
    stars.add_fake_constellation(n_stars=8, mag=9.5)
    aca = get_aca_catalog(
        **mod_std_info(n_fid=0, n_guide=8, obsid=50000),
        stars=stars,
        dark=DARK40,
        raise_exc=True
    )
    acar = ACAReviewTable(aca)
    acar.check_guide_count()
    assert len(acar.messages) == 1
    msg = acar.messages[0]
    assert msg["category"] == "critical"
    assert "ER count of 9th" in msg["text"]


def test_guide_count_er2():
    # This configuration should have not enough stars overall
    stars = StarsTable.empty()
    stars.add_fake_constellation(n_stars=3, mag=[8.5, 8.5, 8.5])
    aca = get_aca_catalog(
        **mod_std_info(n_fid=0, n_guide=8, obsid=50000),
        stars=stars,
        dark=DARK40,
        raise_exc=True
    )
    acar = ACAReviewTable(aca)
    acar.check_guide_count()
    assert acar.messages == [
        {"text": "ER count of guide stars 3.00 < 6.0", "category": "critical"},
        {"text": "ER with 3 guides but 8 were requested", "category": "caution"},
    ]


def test_guide_count_er3():
    # And this configuration should have about the bare minumum (of course better
    # to do this with programmatic instead of fixed checks... TODO)
    stars = StarsTable.empty()
    stars.add_fake_constellation(n_stars=6, mag=[8.5, 8.5, 8.5, 9.9, 9.9, 9.9])
    aca = get_aca_catalog(
        **mod_std_info(obsid=50000, n_fid=0, n_guide=8),
        stars=stars,
        dark=DARK40,
        raise_exc=True
    )
    acar = ACAReviewTable(aca)
    acar.check_guide_count()
    assert acar.messages == [
        {"text": "ER with 6 guides but 8 were requested", "category": "caution"}
    ]


def test_guide_count_er4():
    # This configuration should not warn with too many really bright stars
    # (allowed to have 3 stars brighter than 6.1)
    stars = StarsTable.empty()
    stars.add_fake_constellation(n_stars=3, mag=8.0)
    stars.add_fake_star(yang=100, zang=100, mag=6.0)
    stars.add_fake_star(yang=1000, zang=-1000, mag=6.0)
    stars.add_fake_star(yang=-2000, zang=-2000, mag=6.0)
    aca = get_aca_catalog(
        **mod_std_info(obsid=50000, n_fid=0, n_guide=8),
        stars=stars,
        dark=DARK40,
        raise_exc=True
    )
    acar = ACAReviewTable(aca)
    acar.check_guide_count()
    assert acar.messages == [
        {"text": "ER with 6 guides but 8 were requested", "category": "caution"}
    ]


def test_include_exclude():
    """Test INFO statement for explicitly included/excluded entries"""
    stars = StarsTable.empty()
    stars.add_fake_constellation(n_stars=8, mag=np.linspace(7.0, 8.75, 8))
    aca = get_aca_catalog(
        **mod_std_info(obsid=50000, n_fid=0, n_acq=5, n_guide=8),
        stars=stars,
        dark=DARK40,
        exclude_ids_acq=100,
        include_ids_guide=107,
        exclude_ids_guide=[100, 101],
        include_ids_acq=[106, 107],
        include_halfws_acq=[140, 120],
        raise_exc=True
    )
    acar = ACAReviewTable(aca)
    acar.check_include_exclude()
    assert acar.messages == [
        {
            "category": "info",
            "text": "included acq ID(s): [106, 107] halfwidths(s): [140, 120]",
        },
        {"category": "info", "text": "excluded acq ID(s): 100"},
        {"category": "info", "text": "included guide ID(s): 107"},
        {"category": "info", "text": "excluded guide ID(s): [100, 101]"},
    ]


def test_guide_count_er5():
    # This configuration should warn with too many bright stars
    # (has > 3.0 stars brighter than 5.5
    stars = StarsTable.empty()
    stars.add_fake_constellation(n_stars=4, mag=5.4)
    stars.add_fake_star(yang=1000, zang=1000, mag=8.0)
    stars.add_fake_star(yang=-1000, zang=1000, mag=8.0)
    aca = get_aca_catalog(
        **mod_std_info(obsid=50000, n_fid=0, n_guide=8),
        stars=stars,
        dark=DARK40,
        raise_exc=True
    )
    acar = ACAReviewTable(aca)
    acar.check_guide_count()
    assert acar.messages == [
        {"text": "ER with more than 3 stars brighter than 5.5.", "category": "caution"},
        {"text": "ER with 6 guides but 8 were requested", "category": "caution"},
    ]


def test_guide_count_or():
    """Test the check that an OR has enough fractional guide stars by guide_count"""
    stars = StarsTable.empty()
    stars.add_fake_constellation(n_stars=5, mag=[7.0, 7.0, 10.3, 10.3, 10.3])
    aca = get_aca_catalog(
        **mod_std_info(n_fid=3, n_guide=5, obsid=1),
        stars=stars,
        dark=DARK40,
        raise_exc=True
    )
    acar = ACAReviewTable(aca)
    acar.check_guide_count()
    assert acar.messages == [
        {"text": "OR count of guide stars 2.00 < 4.0", "category": "critical"},
        {"text": "OR with 2 guides but 5 were requested", "category": "caution"},
    ]


def test_ok_number_bright_guide_stars():
    # This configuration should not warn with too many really bright stars
    # (allowed to have 1 stars brighter than 5.5)
    stars = StarsTable.empty()
    stars.add_fake_constellation(n_stars=3, mag=8.0)
    stars.add_fake_star(yang=100, zang=100, mag=5.4)
    aca = get_aca_catalog(
        **mod_std_info(n_fid=3, n_guide=5, obsid=1),
        stars=stars,
        dark=DARK40,
        raise_exc=True
    )
    acar = ACAReviewTable(aca)
    acar.check_guide_count()
    assert acar.messages == [
        {"text": "OR with 4 guides but 5 were requested", "category": "caution"}
    ]


def test_too_many_bright_stars():
    # This configuration should warn with too many bright stars
    # (has > 1.0 stars brighter than 5.5
    stars = StarsTable.empty()
    stars.add_fake_constellation(n_stars=3, mag=8.0)
    stars.add_fake_star(yang=1000, zang=1000, mag=5.4)
    stars.add_fake_star(yang=-1000, zang=1000, mag=5.4)
    aca = get_aca_catalog(
        **mod_std_info(n_fid=3, n_guide=5, obsid=1),
        stars=stars,
        dark=DARK40,
        raise_exc=True
    )
    acar = ACAReviewTable(aca)
    acar.check_guide_count()
    assert len(acar.messages) == 1
    msg = acar.messages[0]
    assert msg["category"] == "caution"
    assert "OR with more than 1 stars brighter than 5.5." in msg["text"]


def test_low_guide_count():
    """Test that a 3.5 to 4.0 guide_count observation gets a critical warning
    on guide_count if man_angle_next > 5 (no creep-away)."""
    # Set a scenario with guide_count in the 3.5 to 4.0 range and confirm a
    # critical warning.
    stars = StarsTable.empty()
    stars.add_fake_constellation(n_stars=5, mag=[7.0, 7.0, 7.0, 10.2, 10.3])
    aca = get_aca_catalog(
        **mod_std_info(n_fid=3, n_guide=5, obsid=1),
        stars=stars,
        dark=DARK40,
        raise_exc=True
    )
    acar = ACAReviewTable(aca)
    # Confirm the guide_count is in the range we want for the test to be valid
    assert acar.guide_count <= 4.0 and acar.guide_count > 3.5
    assert acar.man_angle_next > 5
    acar.check_guide_count()
    assert acar.messages == [
        {"text": "OR count of guide stars 3.65 < 4.0", "category": "critical"},
        {"text": "OR with 4 guides but 5 were requested", "category": "caution"},
    ]


def test_low_guide_count_creep_away():
    """Test that a 3.5 to 4.0 guide_count observation does not get a critical warning
    on guide_count if man_angle_next <= 5 (creep-away)."""
    # Set a scenario with guide_count in the 3.5 to 4.0 range but with
    # a creep away (maneuver angle <= 5), and confirm that is just a warning
    # (not critical).
    stars = StarsTable.empty()
    stars.add_fake_constellation(n_stars=5, mag=[7.0, 7.0, 7.0, 10.2, 10.3])
    aca = get_aca_catalog(
        **mod_std_info(n_fid=3, n_guide=5, obsid=1, man_angle_next=5.0),
        stars=stars,
        dark=DARK40,
        raise_exc=True
    )
    acar = ACAReviewTable(aca)
    # Confirm the guide_count is in the range we want for the test to be valid
    assert acar.guide_count <= 4.0 and acar.guide_count > 3.5
    acar.check_guide_count()
    assert acar.messages == [
        {"text": "OR count of guide stars 3.65 < 4.0", "category": "warning"},
        {"text": "OR with 4 guides but 5 were requested", "category": "caution"},
    ]


def test_reduced_dither_low_guide_count():
    """Test that a 3.5 to 4.0 guide_count observation without dynamic background
    in use (dyn_bgd_n_faint == 0) does not get a dither critical warning for 4x4 arcsec dither.
    """
    # Set a scenario with guide_count in the 3.5 to 4.0 range
    # dither == 4x4 to get no warnings
    stars = StarsTable.empty()
    stars.add_fake_constellation(n_stars=5, mag=[7.0, 7.0, 7.0, 10.2, 10.3])
    aca = get_aca_catalog(
        **mod_std_info(n_fid=3, n_guide=5, obsid=1, dyn_bgd_n_faint=0, dither=(4, 4)),
        stars=stars,
        dark=DARK40,
        raise_exc=True
    )
    acar = ACAReviewTable(aca)

    # Confirm the guide_count is in the range we want for the test to be valid
    assert acar.guide_count <= 4.0 and acar.guide_count > 3.5

    # Run the dither check
    acar.check_dither()
    assert len(acar.messages) == 0


def test_get_t_ccds_bonus_1():
    mags = [1, 10, 2, 11, 3, 4]
    t_ccd = 10

    # Temps corresponding to two faintest stars are smaller.
    t_ccds = get_t_ccds_bonus(mags, t_ccd, dyn_bgd_n_faint=2, dyn_bgd_dt_ccd=-1)
    assert np.all(t_ccds == [10, 9, 10, 9, 10, 10])

    # Temps corresponding to three faintest stars are smaller.
    t_ccds = get_t_ccds_bonus(mags, t_ccd, dyn_bgd_n_faint=3, dyn_bgd_dt_ccd=-1)
    assert np.all(t_ccds == [10, 9, 10, 9, 10, 9])

    # Temps corresponding to just the three faintest stars are smaller because of the
    # minimum number of anchor stars = 3.
    t_ccds = get_t_ccds_bonus(mags, t_ccd, dyn_bgd_n_faint=4, dyn_bgd_dt_ccd=-1)
    assert np.all(t_ccds == [10, 9, 10, 9, 10, 9])

    t_ccds = get_t_ccds_bonus(mags, t_ccd, dyn_bgd_n_faint=0, dyn_bgd_dt_ccd=-1)
    assert np.all(t_ccds == [10, 10, 10, 10, 10, 10])


def test_get_t_ccds_bonus_min_anchor():
    mags = [1, 10, 2]
    t_ccd = 10
    t_ccds = get_t_ccds_bonus(mags, t_ccd, dyn_bgd_n_faint=2, dyn_bgd_dt_ccd=-1)
    assert np.all(t_ccds == [10, 10, 10])

    t_ccds = get_t_ccds_bonus(mags, t_ccd, dyn_bgd_n_faint=4, dyn_bgd_dt_ccd=-1)
    assert np.all(t_ccds == [10, 10, 10])


def test_get_t_ccds_bonus_small_catalog():
    mags = [1]
    t_ccd = 10
    t_ccds = get_t_ccds_bonus(mags, t_ccd, dyn_bgd_n_faint=2, dyn_bgd_dt_ccd=-1)
    assert np.all(t_ccds == [10])


def test_not_reduced_dither_low_guide_count():
    """Test that a 3.5 to 4.0 guide_count observation without dynamic background
    in use (dyn_bgd_n_faint == 0) gets a dither critical warning for 8x8 arcsec dither.
    """
    # Set a scenario with guide_count in the 3.5 to 4.0 range and
    # implicit dyn_bgd_n_faint=0 and dither > 4x4 to get a critical warning
    stars = StarsTable.empty()
    stars.add_fake_constellation(n_stars=5, mag=[7.0, 7.0, 7.0, 10.2, 10.3])
    aca = get_aca_catalog(
        **mod_std_info(n_fid=3, n_guide=5, obsid=1, dyn_bgd_n_faint=0, dither=(8, 8)),
        stars=stars,
        dark=DARK40,
        raise_exc=True
    )
    acar = ACAReviewTable(aca)

    # Confirm the guide_count is in the range we want for the test to be valid
    assert acar.guide_count <= 4.0 and acar.guide_count > 3.5

    # Run the dither check
    acar.check_dither()
    assert acar.messages == [
        {"text": "guide_count 3.65 and dither > 4x4 arcsec", "category": "critical"}
    ]


def test_not_reduced_dither_low_guide_count_dyn_bgd():
    """Test that a 3.5 to 4.0 guide_count observation with dynamic background
    in use (dyn_bgd_n_faint > 0) does not get a dither critical warning for 8x8 arcsec dither.
    """
    # Set a scenario with guide_count in the 3.5 to 4.0 range with dither
    # 8x8 but dynamic background running (inferred from dyn_bgd_n_faint > 0)
    # This has a fainter last star, as dyn_bgd_n_faint=1 changes the guide_count
    # and the test is intended to get into the 3.5 to 4.0 guide count range.
    stars = StarsTable.empty()
    stars.add_fake_constellation(n_stars=5, mag=[7.0, 7.0, 7.0, 10.2, 10.8])
    aca = get_aca_catalog(
        **mod_std_info(n_fid=3, n_guide=5, obsid=1, dyn_bgd_n_faint=1, dither=(8, 8)),
        stars=stars,
        dark=DARK40,
        raise_exc=True
    )
    acar = ACAReviewTable(aca)
    # Confirm the guide_count is in the range we want for the test to be valid
    assert acar.guide_count <= 4.0 and acar.guide_count > 3.5
    acar.check_dither()
    assert len(acar.messages) == 0


def test_pos_err_on_guide():
    """Test the check that no guide star has large POS_ERR"""
    stars = StarsTable.empty()
    stars.add_fake_star(id=100, yang=100, zang=-200, POS_ERR=2010, mag=8.0)
    stars.add_fake_star(
        id=101, yang=0, zang=500, mag=8.0, POS_ERR=1260
    )  # Just over warning
    stars.add_fake_star(
        id=102, yang=-200, zang=500, mag=8.0, POS_ERR=1240
    )  # Just under warning
    stars.add_fake_star(
        id=103, yang=500, zang=500, mag=8.0, POS_ERR=1260
    )  # Not selected

    aca = get_aca_catalog(
        **mod_std_info(n_fid=0),
        stars=stars,
        dark=DARK40,
        raise_exc=True,
        include_ids_guide=[100, 101]
    )  # Must force 100, 101, pos_err too big

    acar = ACAReviewTable(aca)

    # 103 not selected because pos_err > 1.25 arcsec
    assert acar.guides["id"].tolist() == [100, 101, 102]

    # Run pos err checks
    for guide in aca.guides:
        acar.check_pos_err_guide(guide)

    assert len(acar.messages) == 2
    msg = acar.messages[0]
    assert msg["category"] == "critical"
    assert "Guide star 100 POS_ERR 2.01" in msg["text"]

    msg = acar.messages[1]
    assert msg["category"] == "warning"
    assert "Guide star 101 POS_ERR 1.26" in msg["text"]


def test_guide_overlap():
    stars = StarsTable.empty()
    stars.add_fake_star(id=1, mag=8, row=50, col=-50)
    stars.add_fake_constellation(n_stars=7, mag=8.5)
    stars.add_fake_star(id=2, mag=8, row=60, col=-50)
    aca = get_aca_catalog(
        **mod_std_info(n_fid=0, n_guide=8),
        obsid=40000,
        stars=stars,
        dark=DARK40,
        raise_exc=True,
        include_ids_guide=[1, 2]
    )
    assert 2 in aca.guides["id"]
    assert 1 in aca.guides["id"]
    acar = aca.get_review_table()
    acar.run_aca_review()
    assert len(acar.messages) == 2
    assert (
        acar.messages[0]["text"]
        == "Overlapping track index (within 12 pix) idx [1] and idx [8]"
    )


def test_guide_edge_check():
    stars = StarsTable.empty()
    dither = 8
    row_lim = CCD["row_max"] - CCD["row_pad"] - CCD["window_pad"] - dither / 5
    col_lim = -(CCD["col_max"] - CCD["col_pad"] - CCD["window_pad"] - dither / 5)

    # Set positions just below or above CCD['guide_extra_pad'] in row / col
    # Note that the stars are offset from each other so they don't fail the test
    # for overlapping tracked objects.
    stars.add_fake_star(id=1, mag=8, row=row_lim - 2.9, col=0)
    stars.add_fake_star(id=2, mag=8, row=row_lim - 3.1, col=100)
    stars.add_fake_star(id=3, mag=8, row=row_lim - 5.1, col=200)
    stars.add_fake_star(id=4, mag=8, row=0, col=col_lim + 2.9)
    stars.add_fake_star(id=5, mag=8, row=100, col=col_lim + 3.1)
    stars.add_fake_star(id=6, mag=8, row=200, col=col_lim + 5.1)

    stars.add_fake_constellation(n_stars=6, mag=8.5)

    aca = get_aca_catalog(
        **mod_std_info(n_fid=0, n_guide=8),
        obsid=40000,
        stars=stars,
        dark=DARK40,
        raise_exc=True,
        include_ids_guide=np.arange(1, 7)
    )
    acar = ACAReviewTable(aca)
    acar.check_catalog()

    assert acar.messages == [
        {
            "text": "Less than 5.0 pix edge margin row lim 495.4 val 492.3 delta 3.1",
            "category": "info",
            "idx": 5,
        },
        {
            "text": "Less than 5.0 pix edge margin col lim -502.4 val -499.3 delta 3.1",
            "category": "info",
            "idx": 6,
        },
        {
            "text": "Less than 3.0 pix edge margin row lim 495.4 val 492.5 delta 2.9",
            "category": "critical",
            "idx": 7,
        },
        {
            "text": "Less than 3.0 pix edge margin col lim -502.4 val -499.5 delta 2.9",
            "category": "critical",
            "idx": 8,
        },
        {"text": "included guide ID(s): [1 2 3 4 5 6]", "category": "info"},
    ]


@pytest.mark.parametrize("exp_warn", [False, True])
def test_imposters_on_guide(exp_warn):
    """Test the check for imposters by adding one imposter to a fake star"""
    stars = StarsTable.empty()
    # Add two stars because separate P2 tests seem to break with just one star
    stars.add_fake_constellation(n_stars=5, mag=9.5)
    mag = 8.0
    cnt = mag_to_count_rate(mag)
    stars.add_fake_star(id=110, row=100, col=-200, mag=mag)
    dark_with_badpix = DARK40.copy()
    # Add an imposter that is just over the limit (2.62) for exp_warn=True, else
    # add imposter that is exactly at the limit (2.52) which rounds to 2.5 => no
    # warning.
    scale = 0.105 if exp_warn else 0.1
    dark_with_badpix[100 + 512, -200 + 512] = cnt * scale
    dark_with_badpix[100 + 512, -201 + 512] = cnt * scale
    aca = get_aca_catalog(
        **mod_std_info(n_fid=0, n_guide=8),
        stars=stars,
        dark=dark_with_badpix,
        raise_exc=True
    )
    acar = ACAReviewTable(aca)
    acar.check_imposters_guide(aca.guides.get_id(110))
    if exp_warn:
        assert len(acar.messages) == 1
        msg = acar.messages[0]
        assert msg["category"] == "warning"
        assert msg["text"] == "Guide star imposter offset 2.6, limit 2.5 arcsec"
    else:
        assert len(acar.messages) == 0


def test_bad_star_set():
    bad_id = 1248994952
    star = agasc.get_star(bad_id)
    ra = star["RA"]
    dec = star["DEC"]
    aca = get_aca_catalog(
        **mod_std_info(n_fid=0, att=(ra, dec, 0)),
        dark=DARK40,
        include_ids_guide=[bad_id]
    )
    acar = ACAReviewTable(aca)
    acar.check_catalog()
    assert acar.messages == [
        {
            "text": "Guide star 1248994952 does not meet guide candidate criteria",
            "category": "critical",
            "idx": 5,
        },
        {
            "text": "Star 1248994952 is in proseco bad star set",
            "category": "critical",
            "idx": 5,
        },
        {"text": "OR requested 0 fids but 3 is typical", "category": "caution"},
        {"category": "info", "text": "included guide ID(s): [1248994952]"},
    ]


def test_too_bright_guide_magerr():
    """Test the check for too-bright guide stars within mult*mag_err of 5.2"""
    stars = StarsTable.empty()
    # Add two stars because separate P2 tests seem to break with just one star
    stars.add_fake_star(
        id=100, yang=100, zang=-200, mag=5.4, mag_err=0.11, MAG_ACA_ERR=10
    )
    stars.add_fake_star(id=101, yang=0, zang=500, mag=8.0)
    aca = get_aca_catalog(
        **mod_std_info(n_fid=0), stars=stars, dark=DARK40, raise_exc=True
    )
    acar = ACAReviewTable(aca)
    acar.check_too_bright_guide(aca.guides.get_id(100))
    msg = acar.messages[0]
    assert msg["category"] == "critical"
    assert "2*mag_err of 5.2" in msg["text"]


def test_check_fid_spoiler_score():
    """Test checking fid spoiler score"""
    stars = StarsTable.empty()
    # def add_fake_stars_from_fid(self, fid_id=1, offset_y=0, offset_z=0, mag=7.0,
    #                            id=None, detector='ACIS-S', sim_offset=0):
    stars.add_fake_constellation(n_stars=8)

    # Add a red spoiler on top of fids 1-4 and yellow on top of fid 5
    stars.add_fake_stars_from_fid(fid_id=[1, 2, 3, 4], mag=7 + 2)
    stars.add_fake_stars_from_fid(fid_id=[5], mag=7 + 4.5022222)

    aca = get_aca_catalog(stars=stars, **STD_INFO)

    assert np.all(aca.fids.cand_fids["spoiler_score"] == [4, 4, 4, 4, 1, 0])

    acar = ACAReviewTable(aca)
    acar.check_catalog()
    assert acar.messages == [
        {
            "text": "Fid 1 has red spoiler: star 108 with mag 9.00",
            "category": "critical",
            "idx": 1,
        },
        {
            "text": "Fid 5 has yellow spoiler: star 112 with mag 11.50",
            "category": "warning",
            "idx": 2,
        },
    ]


def test_check_fid_count1():
    """Test checking fid count"""
    stars = StarsTable.empty()
    stars.add_fake_constellation(n_stars=8)

    aca = get_aca_catalog(
        stars=stars, **mod_std_info(detector="HRC-S", sim_offset=40000)
    )
    acar = ACAReviewTable(aca)
    acar.check_catalog()

    assert acar.messages == [
        {"text": "OR has 2 fids but 3 were requested", "category": "critical"}
    ]


def test_check_fid_count2():
    """Test checking fid count"""
    stars = StarsTable.empty()
    stars.add_fake_constellation(n_stars=8)

    aca = get_aca_catalog(stars=stars, **mod_std_info(detector="HRC-S", n_fid=2))
    acar = ACAReviewTable(aca)
    acar.check_catalog()

    assert acar.messages == [
        {"text": "OR requested 2 fids but 3 is typical", "category": "caution"}
    ]


def test_check_guide_geometry():
    """Test the checks of geometry (not all within 2500" not N-2 within 500")"""
    yangs = np.array([1, 0, -1, 0])
    zangs = np.array([0, 1, 0, -1])
    for size, y_offset, z_offset, fail in zip(
        [500, 1200, 1500], [0, 1200, 0], [0, 1200, -1000], [True, True, False]
    ):
        stars = StarsTable.empty()
        for y, z in zip(yangs, zangs):
            stars.add_fake_star(
                yang=y * size + y_offset, zang=z * size + z_offset, mag=7.0
            )
        aca = get_aca_catalog(**STD_INFO, stars=stars, dark=DARK40)
        acar = aca.get_review_table()

        acar.check_guide_geometry()
        if fail:
            assert len(acar.messages) == 1
            msg = acar.messages[0]
            assert msg["category"] == "warning"
            assert "Guide stars all clustered" in msg["text"]
        else:
            assert len(acar.messages) == 0

    # Test for cluster of 3 500" rad stars in a 5 star case
    stars = StarsTable.empty()
    size = 1200
    yangs = np.array([1, 0.90, 1.10, 0, -1])
    zangs = np.array([1, 0.90, 1.10, -1, 0])
    for y, z in zip(yangs, zangs):
        stars.add_fake_star(yang=y * size, zang=z * size, mag=7.0)
    aca = get_aca_catalog(**STD_INFO, stars=stars, dark=DARK40)
    acar = aca.get_review_table()
    acar.check_guide_geometry()

    assert len(acar.messages) == 1
    msg = acar.messages[0]
    assert msg["category"] == "critical"
    assert 'Guide indexes [4, 5, 6] clustered within 500" radius' in msg["text"]

    # Test for cluster of 3 500" rad stars in a 5 star case, but downgrade
    # the warning for the case when the maneuver angle away is <= 5
    # (creep-away).
    stars = StarsTable.empty()
    size = 1200
    yangs = np.array([1, 0.90, 1.10, 0, -1])
    zangs = np.array([1, 0.90, 1.10, -1, 0])
    for y, z in zip(yangs, zangs):
        stars.add_fake_star(yang=y * size, zang=z * size, mag=7.0)

    aca = get_aca_catalog(**STD_INFO, man_angle_next=5.0, stars=stars, dark=DARK40)
    acar = aca.get_review_table()
    acar.check_guide_geometry()

    assert len(acar.messages) == 1
    msg = acar.messages[0]
    assert msg["category"] == "warning"
    assert 'Guide indexes [4, 5, 6] clustered within 500" radius' in msg["text"]


def test_pickle():
    """Test that ACA, guide, acq, and fid catalogs round-trip through pickling.

    Known attributes that do NOT round-trip are below.  None of these are
    required for post-facto catalog evaluation and currently the reporting code
    handles ``stars`` and ``dark``.

    - stars
    - dark
    - aca.fids.acqs

    """
    stars = StarsTable.empty()
    stars.add_fake_constellation(mag=10.0, n_stars=5)
    aca = get_aca_catalog(stars=stars, dark=DARK40, raise_exc=True, **STD_INFO)
    acar = aca.get_review_table()

    acar2 = pickle.loads(pickle.dumps(acar))

    assert repr(acar) == repr(acar2)
    assert repr(acar.acqs.cand_acqs) == repr(acar2.acqs.cand_acqs)

    for cat in None, "acqs", "guides", "fids":
        if cat:
            obj = getattr(acar, cat)
            obj2 = getattr(acar2, cat)
            for event, event2 in zip(obj.log_info["events"], obj2.log_info["events"]):
                assert event == event2
        else:
            obj = acar
            obj2 = acar2

        for attr in ["att", "date", "t_ccd", "man_angle", "dither"]:
            val = getattr(obj, attr)
            val2 = getattr(obj2, attr)
            if isinstance(val, float):
                assert np.isclose(val, val2)
            elif isinstance(val, Quat):
                assert np.allclose(val.q, val2.q)
            else:
                assert val == val2

    # Test that calc_p_safe() gives the same answer, which implicitly tests
    # that the AcqTable.__setstate__ unpickling code has the right (weak)
    # reference to acqs within each AcqProbs object.  This also tests
    # that acqs.p_man_err and acqs.fid_set are the same.
    assert np.isclose(
        acar.acqs.calc_p_safe(), acar2.acqs.calc_p_safe(), atol=0, rtol=1e-6
    )
    assert np.isclose(acar.guide_count, acar2.guide_count, atol=1e-6, rtol=0)
    assert np.isclose(acar.guide_count_9th, acar2.guide_count_9th, atol=1e-6, rtol=0)
    assert np.isclose(acar.acq_count, acar2.acq_count, atol=1e-6, rtol=0)
    assert acar.acqs.fid_set == acar2.acqs.fid_set


def test_copy_deepcopy_pickle():
    """
    Test that copy, deepcopy and pickle all return the expected object which
    is independent of the original (where expected).

    :return:
    """
    aca = get_aca_catalog(**STD_INFO)
    acar = aca.get_review_table()

    def f1(x):
        return pickle.loads(pickle.dumps(x))

    f2 = copy.deepcopy
    f3 = copy.copy

    def f4(x):
        return x.__class__(x)

    for func in (f1, f2, f3, f4):
        acar2 = func(acar)

        # Functional test for #303, mostly just for pickle.
        assert acar2.dark_date == "2017:272"

        for attr in ("acqs", "guides", "fids"):
            val = getattr(acar, attr)
            val2 = getattr(acar2, attr)
            # New table appears the same but is not the same object
            assert repr(val) == repr(val2)
            assert val is not val2

            # Now do the copy func on the lower level table directly
            val2 = func(val)
            assert repr(val) == repr(val2)
            assert val is not val2
