import pytest
import wick


def test_sum_as_string():
    assert wick.sum_as_string(1, 1) == "2"
