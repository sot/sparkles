import ska_helpers

__version__ = ska_helpers.get_version(__package__)

from .core import run_aca_review, ACAReviewTable  # noqa


def test(*args, **kwargs):
    """
    Run py.test unit tests.
    """
    import testr
    return testr.test(*args, **kwargs)
