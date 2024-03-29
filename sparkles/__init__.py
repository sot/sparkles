import ska_helpers

__version__ = ska_helpers.get_version(__package__)

from .aca_check_table import get_t_ccds_bonus  # noqa: F401
from .core import ACAReviewTable, run_aca_review  # noqa: F401


def test(*args, **kwargs):
    """
    Run py.test unit tests.
    """
    import testr

    return testr.test(*args, **kwargs)
