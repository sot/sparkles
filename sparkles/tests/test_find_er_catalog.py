# Licensed under a 3-clause BSD style license - see LICENSE.rst
import warnings
import pytest

import numpy as np
import Ska.Sun
from proseco import get_aca_catalog
from proseco.tests.test_common import mod_std_info
from Quaternion import Quat

from sparkles.find_er_catalog import (
    filter_candidate_stars_on_ccd,
    find_er_catalog,
    get_candidate_stars,
    get_guide_counts,
)


# Known tough field: PKS 0023-26 pointing
@pytest.fixture
def ATT():
    return Quat([0.20668099834, 0.23164729391, 0.002658888173, 0.9505868852])


@pytest.fixture
def DATE():
    return "2021-09-13"


@pytest.fixture
def T_CCD():
    return -8.0


# Get initial catalog at the PKS 0023-26 attitude. Ignore the penalty limit for
# this work.


@pytest.fixture
def KWARGS(ATT, T_CCD, DATE):
    return mod_std_info(
        att=ATT,
        t_ccd=T_CCD,
        date=DATE,
        n_guide=8,
        n_fid=0,
        obsid=99999,
        t_ccd_penalty_limit=999,
    )


@pytest.fixture
def ACA(KWARGS):
    return get_aca_catalog(**KWARGS)


@pytest.fixture
def ATTS(ACA):
    DPITCHES, DYAWS = np.ogrid[-0.01:-3.5:4j, -3.1:3:3j]
    SUN_RA, SUN_DEC = Ska.Sun.position(ACA.date)
    return Ska.Sun.apply_sun_pitch_yaw(
        ACA.att, pitch=DPITCHES, yaw=DYAWS, sun_ra=SUN_RA, sun_dec=SUN_DEC
    )


def test_get_candidate_and_filter_stars(ATT, T_CCD, DATE):
    stars = get_candidate_stars(ATT, T_CCD, date=DATE)
    stars = filter_candidate_stars_on_ccd(ATT, stars)

    count_9th, count_10th, count_all = get_guide_counts(
        stars["MAG_ACA"][stars["guide_mask"]], t_ccd=T_CCD
    )
    assert np.isclose(count_9th, 2.00, atol=0.01)
    assert np.isclose(count_10th, 2.67, atol=0.01)
    assert np.isclose(count_all, 2.25, atol=0.01)


TEST_COLS = [
    "dpitch",
    "dyaw",
    "count_9th",
    "count_10th",
    "count_all",
    "count_ok",
    "n_critical",
    "att",
]


def test_find_er_catalog_minus_2_pitch_bins(ACA, ATTS):
    # Try it all for the bad field near PKS 0023-26
    acar, att_opts = find_er_catalog(ACA, ATTS, alg="pitch_bins")
    # import pprint; pprint.pprint(att_opts[TEST_COLS].pformat_all(), width=100)
    assert acar is att_opts["acar"][8]
    assert att_opts[TEST_COLS].pformat_all() == [
        (
            "dpitch  dyaw count_9th count_10th count_all count_ok n_critical        att"
            "       "
        ),
        (
            "------ ----- --------- ---------- --------- -------- ----------"
            " -----------------"
        ),
        (
            " -0.01 -3.10      4.18       6.00      5.65     True          2  7.67"
            " -25.22 29.3"
        ),
        (
            " -0.01 -0.05      2.00       2.67      2.25    False         --  6.47"
            " -26.05 26.1"
        ),
        (
            " -0.01  3.00      2.62       7.92      5.26    False         --  5.21"
            " -26.82 22.8"
        ),
        (
            " -1.17 -3.10      2.00       9.33      5.92    False         --  8.49"
            " -26.12 29.7"
        ),
        (
            " -1.17 -0.05      0.00       1.23      0.78    False         --  7.23"
            " -27.00 26.4"
        ),
        (
            " -1.17  3.00      0.75       6.87      4.03    False         --  5.91"
            " -27.80 23.1"
        ),
        (
            " -2.34 -3.10      1.89       7.77      5.21    False         --  9.32"
            " -27.02 30.1"
        ),
        (
            " -2.34 -0.05      2.87       8.52      5.97    False         --  8.01"
            " -27.93 26.8"
        ),
        (
            " -2.34  3.00      8.53      13.90     12.67     True          0  6.64"
            " -28.78 23.5"
        ),
        (
            " -3.50 -3.10      2.12      10.01      6.66    False         -- 10.16"
            " -27.91 30.4"
        ),
        (
            " -3.50 -0.05      4.87       9.63      7.50     True         --  8.80"
            " -28.86 27.2"
        ),
        (
            " -3.50  3.00      3.60       9.93      6.38     True         --  7.37"
            " -29.75 23.8"
        ),
    ]


def test_find_er_catalog_minus_2_count_all(ACA, ATTS):
    acar, att_opts = find_er_catalog(ACA, ATTS, alg="count_all")
    # import pprint; pprint.pprint(att_opts[TEST_COLS].pformat_all(), width=100)
    assert acar is att_opts["acar"][8]
    assert att_opts[TEST_COLS].pformat_all() == [
        (
            "dpitch  dyaw count_9th count_10th count_all count_ok n_critical        att"
            "       "
        ),
        (
            "------ ----- --------- ---------- --------- -------- ----------"
            " -----------------"
        ),
        (
            " -0.01 -3.10      4.18       6.00      5.65     True         --  7.67"
            " -25.22 29.3"
        ),
        (
            " -0.01 -0.05      2.00       2.67      2.25    False         --  6.47"
            " -26.05 26.1"
        ),
        (
            " -0.01  3.00      2.62       7.92      5.26    False         --  5.21"
            " -26.82 22.8"
        ),
        (
            " -1.17 -3.10      2.00       9.33      5.92    False         --  8.49"
            " -26.12 29.7"
        ),
        (
            " -1.17 -0.05      0.00       1.23      0.78    False         --  7.23"
            " -27.00 26.4"
        ),
        (
            " -1.17  3.00      0.75       6.87      4.03    False         --  5.91"
            " -27.80 23.1"
        ),
        (
            " -2.34 -3.10      1.89       7.77      5.21    False         --  9.32"
            " -27.02 30.1"
        ),
        (
            " -2.34 -0.05      2.87       8.52      5.97    False         --  8.01"
            " -27.93 26.8"
        ),
        (
            " -2.34  3.00      8.53      13.90     12.67     True          0  6.64"
            " -28.78 23.5"
        ),
        (
            " -3.50 -3.10      2.12      10.01      6.66    False         -- 10.16"
            " -27.91 30.4"
        ),
        (
            " -3.50 -0.05      4.87       9.63      7.50     True         --  8.80"
            " -28.86 27.2"
        ),
        (
            " -3.50  3.00      3.60       9.93      6.38     True         --  7.37"
            " -29.75 23.8"
        ),
    ]


def test_find_er_catalog_minus_2_input_order(ACA, ATTS):
    acar, att_opts = find_er_catalog(ACA, ATTS, alg="input_order")
    # import pprint; pprint.pprint(att_opts[TEST_COLS].pformat_all(), width=100)
    assert acar is att_opts["acar"][8]
    assert att_opts[TEST_COLS].pformat_all() == [
        (
            "dpitch  dyaw count_9th count_10th count_all count_ok n_critical        att"
            "       "
        ),
        (
            "------ ----- --------- ---------- --------- -------- ----------"
            " -----------------"
        ),
        (
            " -0.01 -3.10      4.18       6.00      5.65     True          2  7.67"
            " -25.22 29.3"
        ),
        (
            " -0.01 -0.05      2.00       2.67      2.25    False         --  6.47"
            " -26.05 26.1"
        ),
        (
            " -0.01  3.00      2.62       7.92      5.26    False         --  5.21"
            " -26.82 22.8"
        ),
        (
            " -1.17 -3.10      2.00       9.33      5.92    False         --  8.49"
            " -26.12 29.7"
        ),
        (
            " -1.17 -0.05      0.00       1.23      0.78    False         --  7.23"
            " -27.00 26.4"
        ),
        (
            " -1.17  3.00      0.75       6.87      4.03    False         --  5.91"
            " -27.80 23.1"
        ),
        (
            " -2.34 -3.10      1.89       7.77      5.21    False         --  9.32"
            " -27.02 30.1"
        ),
        (
            " -2.34 -0.05      2.87       8.52      5.97    False         --  8.01"
            " -27.93 26.8"
        ),
        (
            " -2.34  3.00      8.53      13.90     12.67     True          0  6.64"
            " -28.78 23.5"
        ),
        (
            " -3.50 -3.10      2.12      10.01      6.66    False         -- 10.16"
            " -27.91 30.4"
        ),
        (
            " -3.50 -0.05      4.87       9.63      7.50     True         --  8.80"
            " -28.86 27.2"
        ),
        (
            " -3.50  3.00      3.60       9.93      6.38     True         --  7.37"
            " -29.75 23.8"
        ),
    ]


def test_find_er_catalog_fails(ATT, DATE, ATTS):
    """Test a catalog that will certainly fail at +10 degC"""
    kwargs = mod_std_info(
        att=ATT,
        t_ccd=+10,
        date=DATE,
        n_guide=8,
        n_fid=0,
        obsid=99999,
        t_ccd_penalty_limit=999,
    )

    with warnings.catch_warnings():
        # Ignore warning about grid_model clipping t_ccd
        warnings.filterwarnings("ignore", module=r".*star_probs.*")
        warnings.filterwarnings("ignore", message=r".*interpolating MAXMAGs table.*")
        aca = get_aca_catalog(**kwargs)
        acar, att_opts = find_er_catalog(aca, ATTS, alg="input_order")
    assert acar is None
    assert not np.any(att_opts["count_ok"])
