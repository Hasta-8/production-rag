import sys
from pathlib import Path

import pytest

TESTS = Path(__file__).parent
sys.path.insert(0, str(TESTS))  # make tests/helpers.py importable


@pytest.fixture
def fixtures_dir() -> Path:
    return TESTS / "fixtures"
