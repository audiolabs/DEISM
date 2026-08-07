"""Smoke tests for the native components shipped in binary distributions."""

from deism import libroom_deism
from deism.count_reflections_wrapper import (
    CPP_COUNTING_AVAILABLE,
    count_reflections_cpp,
)


def test_native_components_import_and_execute():
    assert libroom_deism is not None
    assert CPP_COUNTING_AVAILABLE is True
    assert count_reflections_cpp(0, (3.0, 4.0, 5.0), 343.0, 1.0) == 1
