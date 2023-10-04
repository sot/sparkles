import agasc
import pytest


@pytest.fixture(autouse=True)
def use_fixed_chandra_models(monkeypatch):
    monkeypatch.setenv("CHANDRA_MODELS_DEFAULT_VERSION", "3.48")


@pytest.fixture(autouse=True)
def do_not_use_agasc_supplement(monkeypatch):
    monkeypatch.setenv(agasc.SUPPLEMENT_ENABLED_ENV, "False")


# By default test with the latest AGASC version available including release candidates
@pytest.fixture(autouse=True)
def proseco_agasc_rc(monkeypatch):
    agasc_file = agasc.get_agasc_filename("proseco_agasc_*", allow_rc=True)
    monkeypatch.setenv("AGASC_HDF5_FILE", agasc_file)


@pytest.fixture()
def proseco_agasc_1p7(monkeypatch):
    agasc_file = agasc.get_agasc_filename("proseco_agasc_*", version="1p7")
    monkeypatch.setenv("AGASC_HDF5_FILE", agasc_file)
