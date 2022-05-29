import os
import numpy as np

import agasc
from sparkles.yoshi import run_one_yoshi


def test_run_one_yoshi(monkeypatch):
    """Regression test a single run for a real obsid"""
    monkeypatch.setenv(agasc.SUPPLEMENT_ENABLED_ENV, 'False')
    request = {
        'obsid': 20562,
        'chip_id': 3,
        'chipx': 970.0,
        'chipy': 975.0,
        'dec_targ': 66.35,
        'detector': 'ACIS-I',
        'dither_y': 8,
        'dither_z': 8,
        'focus_offset': 0,
        'man_angle': 175.0,
        'obs_date': '2019:174:00:32:23.789',
        'offset_y': -2.3,
        'offset_z': 3.0,
        'ra_targ': 239.06125,
        'roll_targ': 197.12,
        'sim_offset': 0,
        't_ccd': -9.1}

    expected = {'ra_aca': 238.96459762180638,
                'dec_aca': 66.400811774068146,
                'roll_aca': 197.20855489084187,
                'n_critical': 2,
                'n_warning': 1,
                'n_caution': 0,
                'n_info': 1,
                'P2': 1.801521445484349,
                'guide_count': 3.8577301426624357}

    actual = run_one_yoshi(**request, dyn_bgd_n_faint=0)

    for key in expected:
        val = expected[key]
        val2 = actual[key]
        if isinstance(val, float):
            assert np.isclose(val, val2, atol=1e-3)
        else:
            assert val == val2
