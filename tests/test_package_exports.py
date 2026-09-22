"""The public API: every name in ``__all__`` resolves, and nothing else leaks.

Research modules that are not part of the release (the benchmark simulator,
local-FDR layer, dip/split confirmation, ``call_dmrs``) must not be importable
from the package, so a stale import fails loudly instead of running retired
code.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Python"))

import eiger2  # noqa: E402

SUPPORTED = {
    "__version__",
    "CallerProfile",
    "CandidateThresholdError",
    "LEGACY",
    "NEWER",
    "PermutationScanError",
    "RegionFdrConfig",
    "RegionFdrRun",
    "annotate_axes",
    "call_coordinated_regions",
    "coordination_fdr",
}


def test_star_import_succeeds_and_every_name_in_all_resolves():
    namespace: dict = {}
    exec("from eiger2 import *", namespace)
    for name in eiger2.__all__:
        assert name in namespace, f"{name!r} in eiger2.__all__ did not import"
        assert getattr(eiger2, name) is namespace[name]


def test_public_api_is_exactly_the_supported_workflow():
    assert set(eiger2.__all__) == SUPPORTED


@pytest.mark.parametrize("module", ["eiger2.core", "eiger2.coordination",
                                    "eiger2.axis", "eiger2.io"])
def test_module_star_imports_resolve(module):
    mod = importlib.import_module(module)
    namespace: dict = {}
    exec(f"from {module} import *", namespace)
    for name in mod.__all__:
        assert name in namespace, f"{name!r} in {module}.__all__ did not import"


@pytest.mark.parametrize("module", ["eiger2.benchmark", "eiger2.probability",
                                    "eiger2.confirm", "eiger2.layouts",
                                    "eiger2.manifest_utils"])
def test_retired_research_modules_are_not_shipped(module):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)


@pytest.mark.parametrize("name", ["call_dmrs", "region_fdr", "region_fdr_result",
                                  "deflate", "estimate_latent_basis",
                                  "main_region_table", "local_fdr",
                                  "confirm_regions"])
def test_retired_functions_are_gone(name):
    import eiger2.core as core

    assert not hasattr(eiger2, name)
    assert not hasattr(core, name)


def test_default_import_does_not_load_optional_dependencies():
    code = (
        "import sys; import eiger2, eiger2.cli, eiger2.coordination, eiger2.axis; "
        "bad = [m for m in ('numba', 'sklearn', 'diptest', 'matplotlib') "
        "if m in sys.modules]; print(','.join(bad))"
    )
    import os
    import subprocess

    root = str(Path(__file__).resolve().parents[1] / "Python")
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True,
        env={**os.environ, "PYTHONPATH": root},
    )
    assert result.stdout.strip() == ""


def test_version_has_one_source():
    """pyproject reads the version from eiger2/__init__.py; the citation and
    the CLI must agree with it."""

    import re

    root = Path(__file__).resolve().parents[1]
    pyproject = (root / "pyproject.toml").read_text()
    assert 'dynamic = ["version"]' in pyproject
    assert 'version = { attr = "eiger2.__version__" }' in pyproject
    assert not re.search(r"^version\s*=\s*\"", pyproject, flags=re.M)
    citation = (root / "CITATION.cff").read_text()
    assert f'version: "{eiger2.__version__}"' in citation
    assert "__version__" in (root / "Makefile").read_text()


def test_cli_reports_the_package_version(capsys):
    from eiger2 import cli

    with pytest.raises(SystemExit) as stop:
        cli.main(["--version"])
    assert stop.value.code == 0
    assert capsys.readouterr().out.strip() == f"eiger {eiger2.__version__}"
