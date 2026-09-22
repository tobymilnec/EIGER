"""Tests for ``tools/check_wheel.py``, the wheel targets' audit.

The helper decides whether ``make wheel-check`` and ``make wheel-gate`` may
report success, so the failures it exists to catch need covering directly:
paper material riding along in the archive, a missing licence or entry point,
a wrong version or platform tag, a retired research module or an unwanted
default dependency coming back, and -- the reason the gate exists at all --
an environment whose ``eiger2`` could be satisfied by the source checkout.

``contents`` and ``sdist`` are exercised as subprocesses over synthetic
archives; ``installed`` inspects the interpreter running it, so its path
checks are driven through ``main`` with the checkout itself as the forbidden
directory, and its closing package run is called directly.
"""

from __future__ import annotations

import importlib.util
import io
import re
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

CHECKER = Path(__file__).resolve().parent.parent / "tools" / "check_wheel.py"

ROOT = CHECKER.parent.parent
VERSION = re.search(
    r'^__version__ = "([^"]+)"', (ROOT / "Python" / "eiger2" / "__init__.py")
    .read_text(), flags=re.M,
).group(1)
METADATA_DIR = f"eiger-{VERSION}.dist-info"
METADATA = (
    f"Metadata-Version: 2.4\nName: eiger\nVersion: {VERSION}\n"
    "Requires-Dist: numpy>=1.23\nRequires-Dist: pandas>=1.5\n"
    "Requires-Dist: scipy>=1.9\n"
    'Requires-Dist: numba>=0.57; extra == "legacy"\n'
    'Requires-Dist: scikit-learn>=1.1; extra == "legacy"\n'
    'Requires-Dist: pytest>=7.0; extra == "dev"\n'
)
MODULES = (
    "eiger2/__init__.py", "eiger2/axis.py", "eiger2/cli.py",
    "eiger2/coordination.py", "eiger2/core.py", "eiger2/io.py",
    "functions/processing.py",
)

# The entries a correct wheel carries, reduced to what the audit reads.
GOOD_ENTRIES = {
    **dict.fromkeys(MODULES, ""),
    f"{METADATA_DIR}/METADATA": METADATA,
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
        [sys.executable, str(CHECKER), "--version", VERSION, "contents",
         str(wheel)],
        capture_output=True, text=True,
    )


def test_a_correct_wheel_passes_and_reports_what_it_found(tmp_path):
    result = run_contents(write_wheel(tmp_path))
    assert result.returncode == 0, result.stderr
    assert "eiger2, functions" in result.stdout
    assert "licenses/LICENSE" in result.stdout
    assert "eiger2 = eiger2.cli:main" in result.stdout
    assert "no retired module" in result.stdout


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
    ["eiger-2.1.0-py3-none-any.whl",
     f"eiger-{VERSION}-cp312-cp312-macosx_11_0_arm64.whl"],
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
    assert load_checker().main(
        ["--version", VERSION, "installed", "--forbid", str(repo)]
    ) == 1
    assert "PYTHONPATH" in capsys.readouterr().err


def test_installed_rejects_a_sys_path_entry_inside_a_forbidden_directory(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.syspath_prepend(str(tmp_path / "src"))
    assert load_checker().main(
        ["--version", VERSION, "installed", "--forbid", str(tmp_path)]
    ) == 1
    assert "sys.path" in capsys.readouterr().err


def test_installed_rejects_a_working_directory_inside_a_forbidden_directory(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.chdir(tmp_path)
    assert load_checker().main(
        ["--version", VERSION, "installed", "--forbid", str(tmp_path)]
    ) == 1
    assert "working directory" in capsys.readouterr().err


def test_environment_rejects_a_pythonpath_pointing_into_the_checkout(
    monkeypatch, capsys
):
    """Checked before the build: a checkout on PYTHONPATH also makes pip read
    the editable install's egg-info and skip installing the wheel at all."""
    repo = CHECKER.parent.parent
    monkeypatch.setenv("PYTHONPATH", str(repo / "Python"))
    assert load_checker().main(
        ["--version", VERSION, "environment", "--forbid", str(repo)]
    ) == 1
    assert "PYTHONPATH" in capsys.readouterr().err


def test_environment_accepts_a_pythonpath_that_stays_outside(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    exit_code = load_checker().main(
        ["--version", VERSION, "environment", "--forbid", str(CHECKER.parent.parent)]
    )
    assert exit_code == 0
    assert "FAIL" not in capsys.readouterr().err


@pytest.mark.parametrize(
    "module", ["eiger2/benchmark.py", "eiger2/confirm.py", "eiger2/probability.py",
               "functions/plotting.py"],
)
def test_a_retired_research_module_in_the_wheel_fails(tmp_path, module):
    entries = dict(GOOD_ENTRIES)
    entries[module] = ""
    result = run_contents(write_wheel(tmp_path, entries))
    assert result.returncode == 1
    assert module in result.stderr


def test_a_missing_runtime_module_fails(tmp_path):
    entries = dict(GOOD_ENTRIES)
    del entries["eiger2/coordination.py"]
    result = run_contents(write_wheel(tmp_path, entries))
    assert result.returncode == 1
    assert "eiger2/coordination.py" in result.stderr


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("Requires-Dist: diptest>=0.9\n", "forbidden dependency"),
        ("Requires-Dist: numba>=0.57\n", "legacy extra"),
    ],
)
def test_an_unwanted_default_dependency_fails(tmp_path, line, message):
    entries = dict(GOOD_ENTRIES)
    entries[f"{METADATA_DIR}/METADATA"] = METADATA + line
    result = run_contents(write_wheel(tmp_path, entries))
    assert result.returncode == 1
    assert message in result.stderr


def test_a_legacy_extra_without_its_dependencies_fails(tmp_path):
    entries = dict(GOOD_ENTRIES)
    entries[f"{METADATA_DIR}/METADATA"] = "\n".join(
        line for line in METADATA.splitlines() if "scikit-learn" not in line
    )
    result = run_contents(write_wheel(tmp_path, entries))
    assert result.returncode == 1
    assert "scikit-learn" in result.stderr


SDIST_DOCUMENTS = {
    "PKG-INFO": METADATA, "pyproject.toml": "", "README.md": "",
    "LICENSE": "MIT\n", "CITATION.cff": "", "docs/USER_GUIDE.md": "",
    "examples/make_example_data.py": "",
}


def write_sdist(tmp_path, files=None, name=f"eiger-{VERSION}.tar.gz"):
    root = f"eiger-{VERSION}/"
    if files is None:
        files = {
            **SDIST_DOCUMENTS,
            **{f"Python/{module}": "" for module in MODULES},
        }
    path = tmp_path / name
    with tarfile.open(path, "w:gz") as archive:
        for relative, text in files.items():
            data = text.encode()
            info = tarfile.TarInfo(relative if relative.startswith("eiger-")
                                   else root + relative)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return path


def run_sdist(sdist):
    return subprocess.run(
        [sys.executable, str(CHECKER), "--version", VERSION, "sdist", str(sdist)],
        capture_output=True, text=True,
    )


def test_a_correct_sdist_passes(tmp_path):
    result = run_sdist(write_sdist(tmp_path))
    assert result.returncode == 0, result.stderr
    assert "build inputs" in result.stdout


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"Python/eiger2/benchmark.py": ""}, "retired"),
        ({"tests/fixtures/cohort.tsv": ""}, "fixtures"),
        ({"PKG-INFO": METADATA.replace(VERSION, "0.0.1")}, "Version"),
        ({"PKG-INFO": METADATA + "Requires-Dist: diptest\n"}, "forbidden"),
    ],
)
def test_a_bad_sdist_fails(tmp_path, change, message):
    files = {**SDIST_DOCUMENTS, **{f"Python/{module}": "" for module in MODULES}}
    files.update(change)
    result = run_sdist(write_sdist(tmp_path, files))
    assert result.returncode == 1
    assert message in result.stderr


def test_an_sdist_without_its_build_inputs_fails(tmp_path):
    files = {"PKG-INFO": METADATA,
             **{f"Python/{module}": "" for module in MODULES}}
    result = run_sdist(write_sdist(tmp_path, files))
    assert result.returncode == 1
    assert "pyproject.toml" in result.stderr


def test_an_sdist_without_the_user_guide_fails(tmp_path):
    files = {**SDIST_DOCUMENTS, **{f"Python/{module}": "" for module in MODULES}}
    del files["docs/USER_GUIDE.md"]
    result = run_sdist(write_sdist(tmp_path, files))
    assert result.returncode == 1
    assert "USER_GUIDE" in result.stderr


def test_the_installed_package_run_finds_the_planted_block(capsys):
    """The closing step of ``installed``, run against the checkout here: the
    step itself is independent of where the package came from, and testing it
    separately makes a regression a test failure, not only a gate failure."""

    sys.path.insert(0, str(ROOT / "Python"))
    try:
        assert load_checker().check_run(VERSION) == 0
    finally:
        sys.path.remove(str(ROOT / "Python"))
    assert "3/3 permutation scans" in capsys.readouterr().out


def test_the_installed_package_run_rejects_a_wrong_version(capsys):
    sys.path.insert(0, str(ROOT / "Python"))
    try:
        assert load_checker().check_run("0.0.0") == 1
    finally:
        sys.path.remove(str(ROOT / "Python"))
    assert "software_version" in capsys.readouterr().err
