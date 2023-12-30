# Licensed under a 3-clause BSD style license - see LICENSE.rst
import functools

import razl.checks
from razl.core import Message

from sparkles.core import ACAReviewTable

__all__ = [
    "check_acq_p2",
    "check_bad_stars",
    "check_catalog",
    "check_dither",
    "check_fid_count",
    "check_fid_spoiler_score",
    "check_guide_count",
    "check_guide_fid_position_on_ccd",
    "check_guide_geometry",
    "check_guide_is_candidate",
    "check_guide_overlap",
    "check_imposters_guide",
    "check_include_exclude",
    "check_pos_err_guide",
    "check_too_bright_guide",
]


def acar_check_wrapper(func):
    """Wrapper to call check functions with ACAReviewTable.

    Checks in razl.checks are written to return a list of messages, while the checks
    in sparkles.checks are written to add messages to the ACAReviewTable. This wrapper
    converts the former to the latter.
    """

    @functools.wraps(func)
    def wrapper(acar: ACAReviewTable, *args, **kwargs):
        msgs: list[Message] = func(acar, *args, **kwargs)
        messages = [
            {
                key: val
                for key in ("category", "text", "idx")
                if (val := getattr(msg, key)) is not None
            }
            for msg in msgs
        ]

        acar.messages.extend(messages)

    return wrapper


check_acq_p2 = acar_check_wrapper(razl.checks.check_acq_p2)
check_bad_stars = acar_check_wrapper(razl.checks.check_bad_stars)
check_catalog = acar_check_wrapper(razl.checks.check_catalog)
check_dither = acar_check_wrapper(razl.checks.check_dither)
check_fid_count = acar_check_wrapper(razl.checks.check_fid_count)
check_fid_spoiler_score = acar_check_wrapper(razl.checks.check_fid_spoiler_score)
check_guide_count = acar_check_wrapper(razl.checks.check_guide_count)
check_guide_fid_position_on_ccd = acar_check_wrapper(
    razl.checks.check_guide_fid_position_on_ccd
)
check_guide_geometry = acar_check_wrapper(razl.checks.check_guide_geometry)
check_guide_is_candidate = acar_check_wrapper(razl.checks.check_guide_is_candidate)
check_guide_overlap = acar_check_wrapper(razl.checks.check_guide_overlap)
check_imposters_guide = acar_check_wrapper(razl.checks.check_imposters_guide)
check_include_exclude = acar_check_wrapper(razl.checks.check_include_exclude)
check_pos_err_guide = acar_check_wrapper(razl.checks.check_pos_err_guide)
check_too_bright_guide = acar_check_wrapper(razl.checks.check_too_bright_guide)
