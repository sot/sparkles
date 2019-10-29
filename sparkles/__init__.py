from ._version import get_versions
from .core import run_aca_review, ACAReviewTable  # noqa


def test(*args, **kwargs):
    """
    Run py.test unit tests.
    """
    import testr
    return testr.test(*args, **kwargs)


__version__ = get_versions()['version']
del get_versions
