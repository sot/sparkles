import pytest
import agasc


@pytest.fixture(autouse=True)
def use_fixed_chandra_models(monkeypatch):
    monkeypatch.setenv("CHANDRA_MODELS_DEFAULT_VERSION", "3.48")


@pytest.fixture(autouse=True)
def do_not_use_agasc_supplement(monkeypatch):
    monkeypatch.setenv(agasc.SUPPLEMENT_ENABLED_ENV, "False")
