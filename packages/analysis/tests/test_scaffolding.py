"""Unit tests for the analysis scaffolding (CLI wiring and artefact paths)."""

from __future__ import annotations

from pathlib import Path

from analysis.cli import app
from analysis.paths import (
    artefacts_dir,
    find_repo_root,
    manifest_path,
    run_dir,
    stage_dir,
)
from typer.testing import CliRunner

runner = CliRunner()


def test_help_lists_the_pipeline_stages():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for stage in ("cohort", "fit", "stratify", "drift"):
        assert stage in result.stdout


def test_unimplemented_stage_exits_non_zero():
    # `report` is the last stage still on the planned panel; invoking it must exit non-zero
    # via the _todo stub. (Switch to another planned stage once report is implemented.)
    result = runner.invoke(app, ["report"])
    assert result.exit_code == 1


def test_artefact_paths_compose_under_the_stage_and_run():
    root = Path("/tmp/repo")
    assert artefacts_dir(root) == root / "artefacts"
    assert stage_dir(root, "fit") == root / "artefacts" / "fit"
    run = run_dir(root, "fit", "abc123")
    assert run == root / "artefacts" / "fit" / "abc123"
    assert manifest_path(run) == run / "manifest.json"


def test_find_repo_root_honours_the_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("ANALYSIS_ROOT", str(tmp_path))
    assert find_repo_root() == tmp_path.resolve()


def test_find_repo_root_locates_a_data_directory(monkeypatch, tmp_path):
    monkeypatch.delenv("ANALYSIS_ROOT", raising=False)
    (tmp_path / "data").mkdir()
    nested = tmp_path / "packages" / "analysis"
    nested.mkdir(parents=True)
    assert find_repo_root(nested) == tmp_path.resolve()
