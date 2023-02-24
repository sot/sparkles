import os

import pytest
from proseco import get_aca_catalog

import agasc


from sparkles import run_aca_review
import warnings

warnings.simplefilter("error")

# Do not use the AGASC supplement in testing by default since mags can change
os.environ[agasc.SUPPLEMENT_ENABLED_ENV] = 'False'

KWARGS_48464 = {'att': [-0.51759295, -0.30129397, 0.27093045, 0.75360213],
                'date': '2019:031:13:25:30.000',
                'detector': 'ACIS-S',
                'dither_acq': (7.9992, 7.9992),
                'dither_guide': (7.9992, 7.9992),
                'man_angle': 67.859,
                'n_acq': 8,
                'n_fid': 0,
                'n_guide': 8,
                'obsid': 48464,
                'sim_offset': -3520.0,
                'focus_offset': 0,
                't_ccd_acq': -9.943,
                't_ccd_guide': -9.938}


def test_run_aca_review_function():
    aca = get_aca_catalog(**KWARGS_48464)
    acar = aca.get_review_table()
    acars = [acar]

    exc = run_aca_review(load_name='test_load', report_dir='tmp', acars=acars)


    """
    Test that the behavior of get_roll_intervals is different for ORs and ERs with
    regard to use of offsets.  They are set to arbitrary large values in the test.
    """

    # This uses the catalog at KWARGS_48464, but would really be better as a fully
    # synthetic test
    obs_kwargs = KWARGS_48464.copy()
    # Use these values to override the get_roll_intervals ranges to get more interesting
    # outputs.
    obs_kwargs['target_offset'] = (20 / 60., 30 / 60)  # deg
    aca_er = get_aca_catalog(**obs_kwargs)
    acar_er = aca_er.get_review_table()

    max_roll_dev = 5

    with pytest.warns(FutureWarning):
        er_roll_intervs, er_info = acar_er.get_roll_intervals(
            acar_er.get_candidate_better_stars(),
            roll_dev=max_roll_dev)

 







