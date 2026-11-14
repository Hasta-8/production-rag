import importlib
from importlib.metadata import version

import pytest

SUBPACKAGES = ["ingestion", "indexing", "retrieval", "generation", "agent", "service"]


def test_package_version_matches_metadata():
    import rag

    assert rag.__version__ == version("productionrag")


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackages_import(name):
    importlib.import_module(f"rag.{name}")
