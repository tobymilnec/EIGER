#!/usr/bin/env python3
"""Audit a built EIGER wheel and the environment it was installed into.

    python tools/check_wheel.py environment --forbid DIR [--forbid DIR]
    python tools/check_wheel.py contents  DIST/eiger-2.2.0-py3-none-any.whl
    python tools/check_wheel.py installed --wheel WHEEL --forbid DIR [--forbid DIR]

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
* ``LICENSE`` is present under the distribution metadata;
* the ``eiger2`` console entry point is declared.

``installed`` must be run **by the interpreter of the target environment, from
a working directory outside the source checkout**.  It requires that the
imported package comes from that environment's ``site-packages`` and from the
built wheel, not from a checkout that happens to be on ``sys.path``: a missing
packaged file must fail here rather than being satisfied by the source tree.
``--forbid`` marks a directory that nothing importable may come from; pass the
checkout and, in CI, the workspace.
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
                "material belongs in the eiger_paper repository"
            )
    missing = [pkg for pkg in ALLOWED_TOP_LEVEL if pkg not in top_level]
    if missing:
        return fail(f"the wheel does not contain {missing}")
    if not license_names:
        return fail(f"no LICENSE under {metadata_dir}/")
    if "eiger2 = eiger2.cli:main" not in entry_points:
        return fail(
            "the eiger2 console entry point is not declared in "
            f"{metadata_dir}/entry_points.txt"
        )

    print(f"[wheel] archive: {wheel.name} ({len(names)} entries)")
    print(f"[wheel] top level: {', '.join(top_level)}")
    print(f"[wheel] licence: {license_names[0]}")
    print("[wheel] entry point: eiger2 = eiger2.cli:main")
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

    # Every advertised import, from this environment only.  benchmark is
    # listed explicitly: its paper-only modes moved out, and it must import
    # with no fixture present.
    for name in ("eiger2.benchmark", "eiger2.cli", "functions.load_functions",
                 "functions.scoring"):
        try:
            __import__(name)
        except ImportError as exc:
            return fail(f"{name} does not import from the installed wheel: {exc}")
        imported = Path(sys.modules[name].__file__).resolve()
        if purelib not in imported.parents:
            return fail(f"{name} imported from {imported}, not from {purelib}")
    print("[wheel] eiger2.benchmark, eiger2.cli, functions.* import from the wheel")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="2.2.0", help="expected version")
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
    installed.set_defaults(handler=check_installed)

    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
