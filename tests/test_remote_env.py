"""The runner sets a job's connection env vars only while the job runs."""

from __future__ import annotations

import os

from nenyax.remote import job_env


def test_job_env_is_scoped(monkeypatch):
    monkeypatch.setenv("NENYAX_KEEP", "outer")
    monkeypatch.delenv("NENYAX_TEMP", raising=False)
    with job_env({"NENYAX_KEEP": "inner", "NENYAX_TEMP": "secret"}):
        assert os.environ["NENYAX_KEEP"] == "inner" and os.environ["NENYAX_TEMP"] == "secret"
    assert os.environ["NENYAX_KEEP"] == "outer" and "NENYAX_TEMP" not in os.environ


def test_job_env_restores_after_errors(monkeypatch):
    monkeypatch.delenv("NENYAX_TEMP", raising=False)
    try:
        with job_env({"NENYAX_TEMP": "x"}):
            raise RuntimeError
    except RuntimeError:
        pass
    assert "NENYAX_TEMP" not in os.environ


def test_job_env_none_is_a_no_op():
    before = dict(os.environ)
    with job_env(None):
        pass
    assert dict(os.environ) == before
