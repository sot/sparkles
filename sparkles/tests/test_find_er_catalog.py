# Licensed under a 3-clause BSD style license - see LICENSE.rst

from proseco import get_aca_catalog
from proseco.tests.test_common import mod_std_info
import numpy as np
from Quaternion import Quat
import Ska.Sun

from sparkles.find_er_catalog import (
    get_candidate_stars, find_er_catalog, filter_candidate_stars_on_ccd,
    get_guide_counts, init_quat_from_attitude)
from sparkles.att_utils import (
    get_sun_pitch_yaw, apply_sun_pitch_yaw,
)

# ## Case example: a poor candidate attitude


# Known tough field: PKS 0023-26 pointing
att_pks = Quat([0.20668099834, 0.23164729391, 0.002658888173, 0.9505868852])
date_pks = '2021-09-13'
t_ccd = -8.0
att_pks.equatorial


# Get initial catalog at the PKS 0023-26 attitude. Ignore the penalty limit for
# this work.
kwargs = mod_std_info(att=att_pks, t_ccd=t_ccd, date='2021-09-13', n_guide=8,
                      n_fid=0, obsid=99999, t_ccd_penalty_limit=999)
aca_pks = get_aca_catalog(**kwargs)


def test_get_candidate_and_filter_stars():
    stars = get_candidate_stars(att_pks, t_ccd, date=date_pks)
    stars = filter_candidate_stars_on_ccd(att_pks, stars)

    count_9th, count_10th, count_all = get_guide_counts(
        stars['MAG_ACA'][stars['guide_mask']], t_ccd=-10)
    count_9th.round(2), count_10th.round(2), count_all.round(2)


def test_init_quat_from_attitude():
    # Basic tests for init_quat_from_attitude
    q = init_quat_from_attitude([Quat([0, 1, 2]),
                                 Quat([3, 4, 5])])
    print('From 1-d list of Quat')
    print(q.equatorial)

    q = init_quat_from_attitude([[Quat([0, 1, 2]), Quat([3, 4, 5])]])
    print('From 2-d list of Quat')
    print(q.equatorial)

    q = init_quat_from_attitude([[0, 1, 2], [3, 4, 5]])
    print('From 2-d list of floats')
    print(q.equatorial)

    q = init_quat_from_attitude([[0, 1, 2], [0, 1, 0, 0]])
    print('From heterogenous list of floats')
    print(q.equatorial)


def test_apply_get_sun_pitch_yaw():
    # Test apply and get sun_pitch_yaw with multiple components
    att = apply_sun_pitch_yaw([0, 45, 0], pitch=[0, 10, 20], yaw=[0, 5, 10],
                              sun_ra=0, sun_dec=90)
    pitch, yaw = get_sun_pitch_yaw(att.ra, att.dec, sun_ra=0, sun_dec=90)
    print(pitch)
    print(yaw)


def test_apply_sun_pitch_yaw():
    # Basic test of apply_sun_pitch_yaw
    att = Quat(equatorial=[0, 45, 0])
    att2 = apply_sun_pitch_yaw(att, pitch=10, yaw=0, sun_ra=0, sun_dec=0)
    print('pitch by 10:', att2.ra, att2.dec.round(1))

    att2 = apply_sun_pitch_yaw(att, pitch=0, yaw=10, sun_ra=0, sun_dec=90)
    print('yaw by 10:', att2.ra, att2.dec.round(1))


def test_apply_sun_pitch_yaw_with_grid():
    # Use np.ogrid to make a grid of RA/Dec values (via dpitches and dyaws)
    # See also np.mgrid.
    dpitches, dyaws = np.ogrid[0:-3:2j, -5:5:3j]
    atts = apply_sun_pitch_yaw(att=[0, 45, 0], pitch=dpitches, yaw=dyaws, sun_ra=0, sun_dec=90)
    print(f'{atts.shape=}')
    assert atts.shape == (2, 3)


def test_find_er_catalog_minus_2():
    # Try it all for the bad field near PKS 0023-26
    dpitches, dyaws = np.ogrid[0:-3.5:5j, -3:3:5j]
    sun_ra, sun_dec = Ska.Sun.position(aca_pks.date)
    atts = apply_sun_pitch_yaw(aca_pks.att, pitch=dpitches, yaw=dyaws,
                               sun_ra=sun_ra, sun_dec=sun_dec)

    aca_pks.call_args['t_ccd'] = -2.0
    acar, att_opts = find_er_catalog(aca_pks, atts, alg='pitch_bins')
    print(acar)
    att_opts.pprint_all()
    print(acar.guides.t_ccd)
    print(acar.att.equatorial)

    acar, att_opts = find_er_catalog(aca_pks, atts, alg='count_all')
    att_opts.pprint_all()
    print(acar.guides.t_ccd)
    print(acar.att.equatorial)

    acar, att_opts = find_er_catalog(aca_pks, atts, alg='input_order')
    att_opts.pprint_all()
    print(acar.guides.t_ccd)
    print(acar.att.equatorial)

    aca_pks.call_args['t_ccd'] = -12.0
    acar, att_opts = find_er_catalog(aca_pks, atts, alg='pitch_bins')
    att_opts.pprint_all()
    print(att_opts['acar'][0].messages)

    acar, att_opts = find_er_catalog(aca_pks, atts, alg='count_all')
    att_opts.pprint_all()

    acar, att_opts = find_er_catalog(aca_pks, atts, alg='input_order')
    att_opts.pprint_all()
