#!/usr/bin/env python3
"""Audit built EIGER distributions and the environment a wheel was installed into.

    python tools/check_wheel.py --version V environment --forbid DIR [--forbid DIR]
    python tools/check_wheel.py --version V contents  DIST/eiger-V-py3-none-any.whl
    python tools/check_wheel.py --version V sdist     DIST/eiger-V.tar.gz
    python tools/check_wheel.py --version V installed --wheel WHEEL --forbid DIR \
        [--legacy] [--require-absent MODULE ...]

Used by the ``wheel-check`` and ``wheel-gate`` targets in the root ``Makefile``.
It exists so that a green wheel target means "the artefact that a user would
install was inspected", never merely "an import statement did not raise".

``environment`` runs first, before anything is built or installed, and
refuses to proceed when ``PYTHONPATH`` reaches into a forbidden directory.  A
checkout on ``PYTHONPATH`` does more than shadow an import: pip reads the
``eiger.egg-info`` an editable install leaves there, reports that eiger "is
already installed", and skips the wheel entirely.  Failing up front makes that
a clear refusal rather than a confusing pip message.

``contents`` reads the wheel archive without installing it and requires that

* the file is the expected pure-Python ``eiger-VERSION-py3-none-any.whl``;
* its top level is exactly ``eiger2``, ``functions`` and the distribution
  metadata directory -- no benchmarking, paper, analysis, provenance, script,
  result or fixture material may ride along;
* every runtime module is present and no retired research module came back;
* the metadata declares no dependency the default workflow does not import
  (``numba`` and ``scikit-learn`` only under the ``legacy`` extra, no
  ``diptest``);
* ``LICENSE`` is present under the distribution metadata;
* the ``eiger2`` console entry point is declared.

``sdist`` reads the source archive the same way: one ``eiger-VERSION/`` root,
the build inputs, the runtime modules, and none of the retired modules or
research material. The wheel ``python -m build`` produces is built from this
sdist, so a wheel that passes ``contents`` also shows the sdist is
self-contained.

``installed`` must be run **by the interpreter of the target environment, from
a working directory outside the source checkout**.  It requires that the
imported package comes from that environment's ``site-packages`` and from the
built wheel, not from a checkout that happens to be on ``sys.path``: a missing
packaged file must fail here rather than being satisfied by the source tree.
``--forbid`` marks a directory that nothing importable may come from; pass the
checkout and, in CI, the workspace.  ``--require-absent`` names a module that
must not be importable at all (the optional ``legacy`` dependencies in a
default install); ``--legacy`` additionally imports the legacy covariance code.
It then runs one small permutation-FDR call through the installed package, so
the artefact a user installs is exercised, not only opened.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path

# Top-level archive entries a correct wheel may contain, besides the
# distribution metadata directory.
ALLOWED_TOP_LEVEL = ("eiger2", "functions")

# Directory names from the pre-split tree.  None of these belongs in the
# software wheel; they are named individually so a regression says which one
# came back rather than only "unexpected content".
FORBIDDEN_COMPONENTS = (
    "benchmarking", "paper", "analysis", "provenance",
    "scripts", "results", "fixtures",
)

# Every module the release needs at run time.
REQUIRED_MODULES = (
    "eiger2/__init__.py", "eiger2/axis.py", "eiger2/cli.py",
    "eiger2/coordination.py", "eiger2/core.py", "eiger2/io.py",
    "functions/processing.py",
)

# Research modules removed from the release.  Named so that a regression says
# which one came back.
RETIRED_MODULES = (
    "eiger2/benchmark.py", "eiger2/probability.py", "eiger2/confirm.py",
    "eiger2/layouts.py", "eiger2/manifest_utils.py",
    "functions/load_functions.py", "functions/scoring.py",
    "functions/plotting.py", "functions/DMR_plotting_TMC.R",
)

# Dependencies the default install must not pull in, and the extra that must
# carry the legacy ones.
FORBIDDEN_REQUIREMENTS = ("diptest",)
LEGACY_REQUIREMENTS = ("numba", "scikit-learn")

# Build inputs the sdist must carry to rebuild the wheel on its own, and the
# documents the README links to.
SDIST_REQUIRED = (
    "PKG-INFO", "pyproject.toml", "README.md", "LICENSE", "CITATION.cff",
    "docs/USER_GUIDE.md", "examples/make_example_data.py",
)

# Modules the default workflow must never import.
OPTIONAL_MODULES = ("numba", "sklearn", "diptest", "matplotlib")


def fail(message: str) -> int:
    print(f"[wheel] FAIL: {message}", file=sys.stderr)
    return 1


def check_contents(args: argparse.Namespace) -> int:
    wheel = Path(args.wheel)
    expected_name = f"eiger-{args.version}-py3-none-any.whl"
    if wheel.name != expected_name:
        return fail(
            f"wheel is named {wheel.name!r}, expected {expected_name!r}: the "
            "version or the platform tag is not what this release declares"
        )
    try:
        with zipfile.ZipFile(wheel) as archive:
            names = archive.namelist()
            metadata_dir = f"eiger-{args.version}.dist-info"
            license_names = [
                name for name in names
                if name.startswith(f"{metadata_dir}/")
                and Path(name).name == "LICENSE"
            ]
            entry_points = (
                archive.read(f"{metadata_dir}/entry_points.txt").decode()
                if f"{metadata_dir}/entry_points.txt" in names else ""
            )
            metadata = (
                archive.read(f"{metadata_dir}/METADATA").decode()
                if f"{metadata_dir}/METADATA" in names else None
            )
    except OSError as exc:
        return fail(f"cannot read {wheel}: {exc}")
    except KeyError as exc:
        return fail(f"{wheel} is missing an expected metadata file: {exc}")
    except zipfile.BadZipFile as exc:
        return fail(f"{wheel} is not a readable wheel archive: {exc}")

    if not names:
        return fail(f"{wheel} is empty")

    top_level = sorted({name.split("/")[0] for name in names})
    unexpected = [
        name for name in top_level
        if name not in ALLOWED_TOP_LEVEL and name != metadata_dir
    ]
    if unexpected:
        return fail(
            f"unexpected top-level wheel contents {unexpected}; only "
            f"{list(ALLOWED_TOP_LEVEL)} and {metadata_dir} may ship"
        )
    for name in names:
        offending = [
            part for part in Path(name).parts if part in FORBIDDEN_COMPONENTS
        ]
        if offending:
            return fail(
                f"{name!r} carries {offending[0]!r} into the wheel; that "
                "research material does not ship with the package"
            )
    missing = [pkg for pkg in ALLOWED_TOP_LEVEL if pkg not in top_level]
    if missing:
        return fail(f"the wheel does not contain {missing}")
    problem = _module_problem(set(names), prefix="")
    if problem:
        return fail(problem)
    if metadata is None:
        return fail(f"{metadata_dir}/METADATA is missing")
    problem = _requirement_problem(metadata)
    if problem:
        return fail(problem)
    if not license_names:
        return fail(f"no LICENSE under {metadata_dir}/")
    if "eiger2 = eiger2.cli:main" not in entry_points:
        return fail(
            "the eiger2 console entry point is not declared in "
            f"{metadata_dir}/entry_points.txt"
        )

    print(f"[wheel] archive: {wheel.name} ({len(names)} entries)")
    print(f"[wheel] top level: {', '.join(top_level)}")
    print(f"[wheel] runtime modules present: {len(REQUIRED_MODULES)}; "
          "no retired module")
    print("[wheel] requirements: no diptest; numba and scikit-learn only "
          "under the legacy extra")
    print(f"[wheel] licence: {license_names[0]}")
    print("[wheel] entry point: eiger2 = eiger2.cli:main")
    return 0


def _module_problem(names: set[str], prefix: str) -> str | None:
    """A missing runtime module or a retired one, as a message, else None."""

    missing = [m for m in REQUIRED_MODULES if prefix + m not in names]
    if missing:
        return f"runtime module(s) missing: {missing}"
    retired = [m for m in RETIRED_MODULES if prefix + m in names]
    if retired:
        return f"retired research module(s) shipped: {retired}"
    return None


def _requirement_problem(metadata: str) -> str | None:
    """Check ``Requires-Dist`` lines of wheel METADATA or sdist PKG-INFO."""

    requirements = [
        line.split(":", 1)[1].strip() for line in metadata.splitlines()
        if line.startswith("Requires-Dist:")
    ]

    def name(requirement: str) -> str:
        head = requirement.split(";", 1)[0]
        for stop in "<>=!~[ (":
            head = head.split(stop, 1)[0]
        return head.strip().lower().replace("_", "-")

    for requirement in requirements:
        if name(requirement) in FORBIDDEN_REQUIREMENTS:
            return f"forbidden dependency declared: {requirement!r}"
        if name(requirement) in LEGACY_REQUIREMENTS and (
            'extra == "legacy"' not in requirement
            and "extra == 'legacy'" not in requirement
        ):
            return (f"{requirement!r} is a default dependency; it belongs "
                    "under the legacy extra")
    legacy = {
        name(r) for r in requirements
        if 'extra == "legacy"' in r or "extra == 'legacy'" in r
    }
    missing = [r for r in LEGACY_REQUIREMENTS if r not in legacy]
    if missing:
        return f"the legacy extra does not declare {missing}"
    return None


def check_sdist(args: argparse.Namespace) -> int:
    import tarfile

    sdist = Path(args.sdist)
    expected_name = f"eiger-{args.version}.tar.gz"
    if sdist.name != expected_name:
        return fail(f"sdist is named {sdist.name!r}, expected {expected_name!r}")
    root = f"eiger-{args.version}/"
    try:
        with tarfile.open(sdist, "r:gz") as archive:
            members = [m for m in archive.getmembers() if m.isfile()]
            names = {m.name for m in members}
            pkg_info = (
                archive.extractfile(root + "PKG-INFO").read().decode()
                if root + "PKG-INFO" in names else None
            )
    except (OSError, tarfile.TarError) as exc:
        return fail(f"cannot read {sdist}: {exc}")
    if not names:
        return fail(f"{sdist} is empty")
    outside = sorted(n for n in names if not n.startswith(root))
    if outside:
        return fail(f"entries outside {root}: {outside[:5]}")
    missing = [f for f in SDIST_REQUIRED if root + f not in names]
    if missing:
        return fail(f"the sdist lacks build input(s) {missing}")
    problem = _module_problem(names, prefix=root + "Python/")
    if problem:
        return fail(problem)
    for name in names:
        offending = [
            part for part in Path(name).parts if part in FORBIDDEN_COMPONENTS
        ]
        if offending:
            return fail(f"{name!r} carries {offending[0]!r} into the sdist")
    if f"Version: {args.version}" not in (pkg_info or "").splitlines():
        return fail(f"PKG-INFO does not declare Version: {args.version}")
    problem = _requirement_problem(pkg_info or "")
    if problem:
        return fail(problem)
    print(f"[sdist] archive: {sdist.name} ({len(names)} files under {root})")
    print("[sdist] build inputs, runtime modules and requirements ok")
    return 0


def _forbidden_parent(path: Path, forbidden: list[Path]) -> Path | None:
    for root in forbidden:
        if path == root or root in path.parents:
            return root
    return None


def check_environment(args: argparse.Namespace) -> int:
    """PYTHONPATH only: the one contamination that predates the build."""
    forbidden = [Path(entry).resolve() for entry in args.forbid]
    raw_pythonpath = os.environ.get("PYTHONPATH", "")
    print(f"[wheel] PYTHONPATH={raw_pythonpath!r}")
    for entry in [item for item in raw_pythonpath.split(os.pathsep) if item]:
        root = _forbidden_parent(Path(entry).resolve(), forbidden)
        if root is not None:
            return fail(
                f"PYTHONPATH entry {entry!r} points inside {root}; the checkout "
                "must not be able to satisfy anything the wheel is missing"
            )
    return 0


def check_installed(args: argparse.Namespace) -> int:
    import sysconfig

    forbidden = [Path(entry).resolve() for entry in args.forbid]

    # PYTHONPATH first: it is the cheapest way for a checkout to satisfy a file
    # the wheel forgot, and it must be inspected before anything is imported.
    if check_environment(args):
        return 1
    for entry in [item for item in sys.path if item]:
        root = _forbidden_parent(Path(entry).resolve(), forbidden)
        if root is not None:
            return fail(f"sys.path entry {entry!r} points inside {root}")
    cwd = Path.cwd().resolve()
    root = _forbidden_parent(cwd, forbidden)
    if root is not None:
        return fail(f"the working directory {cwd} is inside {root}")

    try:
        import eiger2
    except ImportError as exc:
        return fail(f"the installed eiger2 does not import: {exc}")

    module = Path(eiger2.__file__).resolve()
    purelib = Path(sysconfig.get_paths()["purelib"]).resolve()
    if purelib not in module.parents:
        return fail(f"eiger2 imported from {module}, not from {purelib}")
    root = _forbidden_parent(module, forbidden)
    if root is not None:
        return fail(f"eiger2 imported from {module}, which is inside {root}")
    if getattr(eiger2, "__version__", None) != args.version:
        return fail(
            f"installed eiger2 reports version "
            f"{getattr(eiger2, '__version__', None)!r}, expected {args.version!r}"
        )
    print(f"[wheel] eiger2 {eiger2.__version__} imported from {module}")
    print(f"[wheel] site-packages: {purelib}")
    print(f"[wheel] working directory: {cwd}")

    from importlib import metadata

    try:
        distribution = metadata.distribution("eiger")
    except metadata.PackageNotFoundError:
        return fail("no installed distribution metadata for 'eiger'")
    if distribution.version != args.version:
        return fail(
            f"distribution metadata reports {distribution.version}, "
            f"expected {args.version}"
        )

    # direct_url.json records what pip actually installed.  An archive URL
    # naming the built wheel is the evidence that this environment came from
    # that file rather than from an index or an editable checkout.
    if args.wheel is not None:
        direct_url_text = distribution.read_text("direct_url.json")
        if direct_url_text is None:
            return fail(
                "the installed distribution has no direct_url.json, so it was "
                "not installed from the built wheel"
            )
        try:
            direct_url = json.loads(direct_url_text)
        except json.JSONDecodeError as exc:
            return fail(f"direct_url.json is not valid JSON: {exc}")
        if "dir_info" in direct_url:
            return fail(
                f"eiger was installed from a directory, not from a wheel: "
                f"{direct_url}"
            )
        url = direct_url.get("url", "")
        if not url.endswith(Path(args.wheel).name):
            return fail(
                f"eiger was installed from {url!r}, not from the built wheel "
                f"{Path(args.wheel).name!r}"
            )
        print(f"[wheel] installed from {url}")

    # Every runtime import, from this environment only.  The legacy code
    # imports numba and scikit-learn, so it is imported only with --legacy.
    modules = ["eiger2.cli", "eiger2.coordination", "eiger2.axis",
               "eiger2.core", "eiger2.io"]
    if args.legacy:
        modules.append("functions.processing")
    for name in modules:
        try:
            __import__(name)
        except ImportError as exc:
            return fail(f"{name} does not import from the installed wheel: {exc}")
        imported = Path(sys.modules[name].__file__).resolve()
        if purelib not in imported.parents:
            return fail(f"{name} imported from {imported}, not from {purelib}")
    print(f"[wheel] {', '.join(modules)} import from the wheel")
    if not args.legacy:
        loaded = [m for m in OPTIONAL_MODULES if m in sys.modules]
        if loaded:
            return fail(f"the default import loaded optional module(s) {loaded}")
        print(f"[wheel] default import loads none of {', '.join(OPTIONAL_MODULES)}")

    import importlib.util

    present = [m for m in args.require_absent if importlib.util.find_spec(m)]
    if present:
        return fail(f"module(s) {present} are installed but must be absent "
                    "from this environment")
    if args.require_absent:
        print(f"[wheel] absent as required: {', '.join(args.require_absent)}")

    return check_run(args.version, legacy=args.legacy)


def check_run(version: str, *, legacy: bool = False) -> int:
    """One small permutation-FDR call through the installed package.

    The data are built here, so no fixture or checkout is needed.  3,200 CpGs
    is above the default profile's 3,002-CpG lag minimum; the planted block
    must come back as a candidate, and all three permutation scans must
    complete.
    """

    import numpy as np
    import pandas as pd

    from eiger2 import LEGACY, NEWER, RegionFdrConfig, call_coordinated_regions

    rng = np.random.default_rng(7)
    n_cpg, n_samples = 3_200, 16
    betas = 0.5 + rng.normal(0.0, 0.05, size=(n_cpg, n_samples))
    betas[1_600:1_612] += 0.15 * rng.normal(0.0, 1.0, size=n_samples)
    betas = np.clip(betas, 0.0, 1.0)
    start = 1_000 + 50 * np.arange(n_cpg)
    locations = pd.DataFrame({"chr": "chr1", "start": start, "end": start + 1})
    for profile in (NEWER, LEGACY) if legacy else (NEWER,):
        run = call_coordinated_regions(
            betas, locations,
            config=RegionFdrConfig(profile=profile, fdr_replicates=3),
        )
        manifest = run.manifest
        if manifest.get("software_version") != version:
            return fail(f"the {profile.name} run reports software_version "
                        f"{manifest.get('software_version')!r}")
        if manifest["permutations"]["completed"] != 3:
            return fail(f"the {profile.name} run completed "
                        f"{manifest['permutations']['completed']} of 3 scans")
        covered = (
            (run.regions["cpg_index_start"] <= 1_605)
            & (run.regions["cpg_index_end"] >= 1_605)
        ).any()
        if not covered:
            return fail(f"the {profile.name} run did not find the planted block")
        print(f"[wheel] {profile.name} run: {len(run.regions)} candidate(s), "
              f"3/3 permutation scans, status {manifest['status']}")
    return 0

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", required=True,
                        help="the version this release declares")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    environment = subparsers.add_parser(
        "environment",
        help="check PYTHONPATH before anything is built or installed",
    )
    environment.add_argument(
        "--forbid", action="append", default=[],
        help="a directory PYTHONPATH may not reach into; repeatable",
    )
    environment.set_defaults(handler=check_environment)

    contents = subparsers.add_parser(
        "contents", help="audit the wheel archive without installing it"
    )
    contents.add_argument("wheel")
    contents.set_defaults(handler=check_contents)

    sdist = subparsers.add_parser(
        "sdist", help="audit the source archive without unpacking it"
    )
    sdist.add_argument("sdist")
    sdist.set_defaults(handler=check_sdist)

    installed = subparsers.add_parser(
        "installed",
        help="check the environment running this interpreter (run it outside "
             "the source checkout)",
    )
    installed.add_argument(
        "--wheel", default=None,
        help="the wheel this environment must have been installed from",
    )
    installed.add_argument(
        "--forbid", action="append", default=[],
        help="a directory nothing importable may come from; repeatable",
    )
    installed.add_argument(
        "--legacy", action="store_true",
        help="the legacy extra is installed: import and run the legacy code too",
    )
    installed.add_argument(
        "--require-absent", action="append", default=[],
        help="a module that must not be importable; repeatable",
    )
    installed.set_defaults(handler=check_installed)

    args = parser.parse_args(argv)
    return args.handler(args)

if __name__ == "__main__":
    raise SystemExit(main())
