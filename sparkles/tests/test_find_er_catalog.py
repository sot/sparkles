# Licensed under a 3-clause BSD style license - see LICENSE.rst

from proseco import get_aca_catalog
from proseco.tests.test_common import mod_std_info
import numpy as np
from Quaternion import Quat
import Ska.Sun

from sparkles.find_er_catalog import (
    get_candidate_stars, find_er_catalog, filter_candidate_stars_on_ccd,
    get_guide_counts, init_quat_from_attitude)


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
