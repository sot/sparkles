__version__ = '4.1'

from .core import run_aca_review, ACAReviewTable


def test(*args, **kwargs):
    """
    Run py.test unit tests.
    """
    import testr
    return testr.test(*args, **kwargs)
