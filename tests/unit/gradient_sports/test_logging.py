"""Tests for logging ownership in reusable library modules."""

from __future__ import annotations

import importlib
import logging
from types import ModuleType

import pytest

from tempoctrl.gradient_sports import event_load, ingest, join, tracking_load


@pytest.mark.parametrize(
    "module",
    [ingest, event_load, tracking_load, join],
    ids=lambda module: module.__name__.rsplit(".", maxsplit=1)[-1],
)
def test_library_import_does_not_configure_global_logging(
    module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep logging configuration at executable boundaries.

    Behavior:
        Reload a reusable data module without configuring the process's
        root logging handlers.

    Input:
        Each library module that emits ingest, join, or output messages,
        with ``logging.basicConfig`` replaced by a recording test
        double.

    Expected:
        Reloading every module makes zero configuration calls while its
        existing root handlers remain unchanged.

    Guards against:
        Imports unexpectedly changing log levels, handlers, or formats
        chosen by pipelines, notebooks, tests, and other applications.
    """
    configuration_calls: list[dict[str, object]] = []

    def record_configuration(**kwargs: object) -> None:
        """Record attempts to configure process-wide logging."""
        configuration_calls.append(kwargs)

    monkeypatch.setattr(logging, "basicConfig", record_configuration)

    original_handlers = list(logging.getLogger().handlers)
    importlib.reload(module)

    assert configuration_calls == []
    assert logging.getLogger().handlers == original_handlers
