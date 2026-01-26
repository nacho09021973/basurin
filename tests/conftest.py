"""Pytest configuration for basurin tests.

This module provides fixtures to prevent sys.modules contamination
between tests, ensuring that MagicMock replacements for the experiment
package don't break subsequent tests that need real imports.
"""
from __future__ import annotations

import sys
from unittest import mock

import pytest


@pytest.fixture(autouse=True)
def _clean_experiment_mocks():
    """Remove MagicMock entries from sys.modules for experiment package.

    Some tests replace experiment.* modules with MagicMock objects without
    cleanup, which breaks subsequent tests that import the real modules.
    This fixture runs before each test and removes any mock entries.
    """
    # Keys to check for mock contamination
    experiment_keys = [k for k in sys.modules if k == "experiment" or k.startswith("experiment.")]

    # Remove any that are MagicMock instances (contamination from other tests)
    for key in experiment_keys:
        module = sys.modules.get(key)
        if isinstance(module, mock.MagicMock):
            del sys.modules[key]

    yield  # Run the test

    # Cleanup after test as well (defensive)
    experiment_keys = [k for k in sys.modules if k == "experiment" or k.startswith("experiment.")]
    for key in experiment_keys:
        module = sys.modules.get(key)
        if isinstance(module, mock.MagicMock):
            del sys.modules[key]
