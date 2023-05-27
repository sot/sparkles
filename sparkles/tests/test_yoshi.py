import numpy as np
import agasc
import pytest
from Quaternion import Quat
from mica.archive.tests.test_cda import HAS_WEB_SERVICES
from sparkles.core import ACAReviewTable

from sparkles.yoshi import (
    get_yoshi_params_from_ocat,
    run_one_yoshi,
    convert_yoshi_to_proseco_params,
)


@pytest.fixture(autouse=True)
def do_not_use_agasc_supplement(monkeypatch):
    """Do not use AGASC supplement in any test"""
    monkeypatch.setenv(agasc.SUPPLEMENT_ENABLED_ENV, "False")


@pytest.mark.skipif(not HAS_WEB_SERVICES, reason="No web services available")
def test_run_one_yoshi():
    """Regression test a single run for a real obsid"""
    request = {
        "obsid": 20562,
        "chip_id": 3,
        "chipx": 970.0,
        "chipy": 975.0,
        "dec_targ": 66.35,
        "detector": "ACIS-I",
        "dither_y": 8,
        "dither_z": 8,
        "focus_offset": 0,
        "man_angle": 175.0,
        "obs_date": "2019:174:00:32:23.789",
        "offset_y": -2.3,
        "offset_z": 3.0,
        "ra_targ": 239.06125,
        "roll_targ": 197.12,
        "sim_offset": 0,
        "t_ccd": -9.1,
        "target_name": "Target name",
    }

    expected = {
        "ra_aca": 238.96459762180638,
        "dec_aca": 66.400811774068146,
        "roll_aca": 197.20855489084187,
        "n_critical": 3,
        "n_warning": 1,
        "n_caution": 0,
        "n_info": 1,
        "P2": 1.801521445484349,
        "guide_count": 3.8577301426624357,
    }

    actual = run_one_yoshi(**request, dyn_bgd_n_faint=0)

    for key in expected:
        val = expected[key]
        val2 = actual[key]
        if isinstance(val, float):
            assert np.isclose(val, val2, atol=1e-3)
        else:
            assert val == val2


@pytest.mark.skipif(not HAS_WEB_SERVICES, reason="No web services available")
def test_get_params():
    params = get_yoshi_params_from_ocat(obsid=8008, obs_date="2022:001")
    exp = {
        "chip_id": 3,
        "chipx": 941.0,
        "chipy": 988.0,
        "dec_targ": 2.3085194444444443,
        "detector": "ACIS-I",
        "dither_y": 7.9992,
        "dither_z": 7.9992,
        "focus_offset": 0,
        "obs_date": "2022:001:00:00:00.000",
        "offset_y": 0.0,
        "offset_z": np.ma.masked,
        "ra_targ": 149.91616666666664,
        "roll_targ": 62.01050867568485,
        "sim_offset": 0,
        "target_name": "C-COSMOS",
    }
    assert_dict_equal(params, exp)

    params_proseco = convert_yoshi_to_proseco_params(
        **params,
        obsid=8008,
        t_ccd=-10,
        man_angle=5.0,
    )
    exp_proseco = {
        "att": Quat([0.15014290, 0.49293780, 0.83025915, 0.21245982]),
        "date": "2022:001:00:00:00.000",
        "detector": "ACIS-I",
        "dither": (7.9992, 7.9992),
        "focus_offset": 0,
        "man_angle": 5.0,
        "n_acq": 8,
        "n_fid": 3,
        "n_guide": 5,
        "obsid": 8008,
        "sim_offset": 0,
        "t_ccd": -10,
        "target_name": "C-COSMOS",
    }
    assert_dict_equal(params_proseco, exp_proseco)


def assert_dict_equal(dict1, dict2):
    assert dict2.keys() == dict1.keys()
    for key in dict2:
        val = dict1[key]
        val_exp = dict2[key]
        if isinstance(val, float):
            assert np.isclose(val, val_exp, atol=1e-8)
        elif val_exp is np.ma.masked:
            assert val is np.ma.masked
        elif isinstance(val_exp, Quat):
            assert str(val) == str(val_exp)
        else:
            assert val == val_exp


@pytest.mark.skipif(not HAS_WEB_SERVICES, reason="No web services available")
def test_get_params_use_cycle():
    """
    Test using the cycle kwarg to get historical obsid configuration but with a recent
    (cycle 21) aimpoint.
    """
    params = get_yoshi_params_from_ocat(obsid=8008, obs_date="2022:001", cycle=21)
    exp = {
        "chip_id": 3,
        "chipx": 970.0,
        "chipy": 975.0,
        "dec_targ": 2.3085194444444443,
        "detector": "ACIS-I",
        "dither_y": 7.9992,
        "dither_z": 7.9992,
        "focus_offset": 0,
        "obs_date": "2022:001:00:00:00.000",
        "offset_y": 0.0,
        "offset_z": np.ma.masked,
        "ra_targ": 149.91616666666664,
        "roll_targ": 62.01050867568485,
        "sim_offset": 0,
        "target_name": "C-COSMOS",
    }
    assert_dict_equal(params, exp)


@pytest.mark.skipif(not HAS_WEB_SERVICES, reason="No web services available")
def test_acar_from_ocat(monkeypatch):
    """Get an AcaReviewTable with minimal information filling in rest from OCAT"""
    monkeypatch.setenv(agasc.SUPPLEMENT_ENABLED_ENV, "False")

    acar = ACAReviewTable.from_ocat(
        obsid=8008, date="2022:001", t_ccd=-10, n_acq=6, target_name="Target name"
    )
    assert acar.obsid == 8008
    assert acar.date == "2022:001:00:00:00.000"
    assert acar.target_name == "Target name"
    assert len(acar.acqs) == 6
    exp = [
        'idx slot    id    type  mag    yang    zang   row    col  ',
        '--- ---- -------- ---- ----- ------- ------- ------ ------',
        '  1    0        1  FID  7.00   919.8  -844.2 -178.7 -164.1',
        '  2    1        5  FID  7.00 -1828.2  1053.8  374.1  216.2',
        '  3    2        6  FID  7.00   385.8  1697.8  -70.8  346.0',
        '  4    3 31983336  BOT  8.64   884.6 -1608.5 -172.3 -317.9',
        '  5    4 31075368  BOT  9.13    57.2   751.5   -5.1  155.5',
        '  6    5 32374896  BOT  9.17  2014.3 -2035.8 -401.4 -405.4',
        '  7    6 31075128  BOT  9.35  -310.8  1200.0   68.9  245.4',
        '  8    7 31463496  BOT  9.46  2035.0  1385.8 -403.5  284.7',
        '  9    0 31076560  ACQ  9.70  -933.4  -354.4  193.0  -66.4',
    ]

    cols = ("idx", "slot", "id", "type", "mag", "yang", "zang")
    assert acar[cols].pformat_all() == exp
