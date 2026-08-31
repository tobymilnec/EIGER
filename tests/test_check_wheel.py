"""Tests for ``tools/check_wheel.py``, the wheel targets' audit.

The helper decides whether ``make wheel-check`` and ``make wheel-gate`` may
report success, so the failures it exists to catch need covering directly:
paper material riding along in the archive, a missing licence or entry point,
a wrong version or platform tag, and -- the reason the gate exists at all --
an environment whose ``eiger2`` could be satisfied by the source checkout.

Only the ``contents`` mode is exercised as a subprocess over synthetic wheels;
``installed`` inspects the interpreter running it, so its path checks are
driven through ``main`` with the checkout itself as the forbidden directory.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

CHECKER = Path(__file__).resolve().parent.parent / "tools" / "check_wheel.py"

VERSION = "2.2.0"
METADATA_DIR = f"eiger-{VERSION}.dist-info"

# The entries a correct 2.2.0 wheel carries, reduced to what the audit reads.
GOOD_ENTRIES = {
    "eiger2/__init__.py": "__version__ = '2.2.0'\n",
    "eiger2/core.py": "",
    "functions/load_functions.py": "",
    f"{METADATA_DIR}/METADATA": "Name: eiger\n",
    f"{METADATA_DIR}/entry_points.txt": "[console_scripts]\neiger2 = eiger2.cli:main\n",
    f"{METADATA_DIR}/licenses/LICENSE": "MIT\n",
}


def write_wheel(tmp_path, entries=None, name=f"eiger-{VERSION}-py3-none-any.whl"):
    path = tmp_path / name
    with zipfile.ZipFile(path, "w") as archive:
        for entry, text in (GOOD_ENTRIES if entries is None else entries).items():
            archive.writestr(entry, text)
    return path


def run_contents(wheel):
    return subprocess.run(
        [sys.executable, str(CHECKER), "contents", str(wheel)],
        capture_output=True, text=True,
    )


def test_a_correct_wheel_passes_and_reports_what_it_found(tmp_path):
    result = run_contents(write_wheel(tmp_path))
    assert result.returncode == 0, result.stderr
    assert "eiger2, functions" in result.stdout
    assert "licenses/LICENSE" in result.stdout
    assert "eiger2 = eiger2.cli:main" in result.stdout


@pytest.mark.parametrize(
    "extra",
    [
        "benchmarking/real_layouts.py",
        "paper/figure_3.py",
        "analysis/cohort.py",
        "provenance/run_log.json",
        "scripts/submit.sh",
        "results/regions.csv",
        "tests/fixtures/wgbs_coordinates.tsv",
    ],
)
def test_paper_material_in_the_archive_fails(tmp_path, extra):
    """The wheel must not regain material that moved to eiger_paper."""
    entries = dict(GOOD_ENTRIES)
    entries[extra] = ""
    result = run_contents(write_wheel(tmp_path, entries))
    assert result.returncode == 1
    assert "FAIL" in result.stderr


def test_an_unexpected_top_level_directory_fails(tmp_path):
    entries = dict(GOOD_ENTRIES)
    entries["old_code/eiger.py"] = ""
    result = run_contents(write_wheel(tmp_path, entries))
    assert result.returncode == 1
    assert "old_code" in result.stderr


def test_a_missing_shipped_package_fails(tmp_path):
    """`--legacy` runs `functions`, so a wheel without it is broken."""
    entries = {
        entry: text for entry, text in GOOD_ENTRIES.items()
        if not entry.startswith("functions/")
    }
    result = run_contents(write_wheel(tmp_path, entries))
    assert result.returncode == 1
    assert "functions" in result.stderr


def test_a_missing_licence_fails(tmp_path):
    entries = {
        entry: text for entry, text in GOOD_ENTRIES.items()
        if "LICENSE" not in entry
    }
    result = run_contents(write_wheel(tmp_path, entries))
    assert result.returncode == 1
    assert "LICENSE" in result.stderr


def test_a_missing_entry_point_fails(tmp_path):
    entries = dict(GOOD_ENTRIES)
    entries[f"{METADATA_DIR}/entry_points.txt"] = "[console_scripts]\n"
    result = run_contents(write_wheel(tmp_path, entries))
    assert result.returncode == 1
    assert "entry point" in result.stderr


@pytest.mark.parametrize(
    "name",
    ["eiger-2.1.0-py3-none-any.whl", "eiger-2.2.0-cp312-cp312-macosx_11_0_arm64.whl"],
)
def test_a_wrong_version_or_platform_tag_fails(tmp_path, name):
    """The wheel must be the pure-Python artefact this release declares."""
    result = run_contents(write_wheel(tmp_path, name=name))
    assert result.returncode == 1
    assert "expected" in result.stderr


def test_a_wheel_that_is_not_an_archive_fails(tmp_path):
    path = tmp_path / f"eiger-{VERSION}-py3-none-any.whl"
    path.write_text("not a zip file")
    result = run_contents(path)
    assert result.returncode == 1
    assert "FAIL" in result.stderr


def load_checker():
    """Import the helper by path; it is a script, not an installed module."""
    spec = importlib.util.spec_from_file_location("check_wheel", CHECKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_installed_rejects_a_pythonpath_pointing_into_the_checkout(monkeypatch, capsys):
    """The failure the gate exists to catch: the checkout on the import path."""
    repo = CHECKER.parent.parent
    monkeypatch.setenv("PYTHONPATH", str(repo / "Python"))
    assert load_checker().main(["installed", "--forbid", str(repo)]) == 1
    assert "PYTHONPATH" in capsys.readouterr().err


def test_installed_rejects_a_sys_path_entry_inside_a_forbidden_directory(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.syspath_prepend(str(tmp_path / "src"))
    assert load_checker().main(["installed", "--forbid", str(tmp_path)]) == 1
    assert "sys.path" in capsys.readouterr().err


def test_installed_rejects_a_working_directory_inside_a_forbidden_directory(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.chdir(tmp_path)
    assert load_checker().main(["installed", "--forbid", str(tmp_path)]) == 1
    assert "working directory" in capsys.readouterr().err


def test_environment_rejects_a_pythonpath_pointing_into_the_checkout(
    monkeypatch, capsys
):
    """Checked before the build: a checkout on PYTHONPATH also makes pip read
    the editable install's egg-info and skip installing the wheel at all."""
    repo = CHECKER.parent.parent
    monkeypatch.setenv("PYTHONPATH", str(repo / "Python"))
    assert load_checker().main(["environment", "--forbid", str(repo)]) == 1
    assert "PYTHONPATH" in capsys.readouterr().err


def test_environment_accepts_a_pythonpath_that_stays_outside(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    exit_code = load_checker().main(
        ["environment", "--forbid", str(CHECKER.parent.parent)]
    )
    assert exit_code == 0
    assert "FAIL" not in capsys.readouterr().err
