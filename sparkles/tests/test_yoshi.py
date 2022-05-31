import numpy as np
import agasc
import pytest
from Quaternion import Quat
from mica.archive.tests.test_cda import HAS_WEB_SERVICES

from sparkles.yoshi import (
    get_yoshi_params_from_ocat,
    run_one_yoshi,
    convert_yoshi_to_proseco_params,
)


@pytest.mark.skipif(not HAS_WEB_SERVICES, reason="No web services available")
def test_run_one_yoshi(monkeypatch):
    """Regression test a single run for a real obsid"""
    monkeypatch.setenv(agasc.SUPPLEMENT_ENABLED_ENV, "False")
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
    }

    expected = {
        "ra_aca": 238.96459762180638,
        "dec_aca": 66.400811774068146,
        "roll_aca": 197.20855489084187,
        "n_critical": 2,
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
        "chipx": 970.0,
        "chipy": 975.0,
        "dec_targ": 2.3085194444444443,
        "detector": "ACIS-I",
        "dither_y": 8,
        "dither_z": 8,
        "focus_offset": 0,
        "obs_date": "2022:001:00:00:00.000",
        "offset_y": 0.0,
        "offset_z": np.ma.masked,
        "ra_targ": 149.91616666666664,
        "roll_targ": 62.01050867568485,
        "sim_offset": 0,
    }
    assert_dict_equal(params, exp)

    params_proseco = convert_yoshi_to_proseco_params(
        **params, obsid=8008, t_ccd=-10, man_angle=5.0
    )
    exp_proseco = {
        "att": Quat([0.15017923, 0.49292814, 0.83025727, 0.21246392]),
        "date": "2022:001:00:00:00.000",
        "detector": "ACIS-I",
        "dither": (8, 8),
        "focus_offset": 0,
        "man_angle": 5.0,
        "n_acq": 8,
        "n_fid": 3,
        "n_guide": 5,
        "obsid": 8008,
        "sim_offset": 0,
        "t_ccd": -10,
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
