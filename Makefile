# EIGER software task interface.
#
# Scope: the installed package only -- lint, software tests, isolated build,
# installed-wheel validation and smoke runs of every documented command over
# the synthetic example.  No cohort data, paper analyses or benchmarks.
#
# Nothing in this file downloads data, touches a cohort, or removes a file it
# did not itself create.  `make` on its own only prints help.
#
# Requires GNU make and a POSIX shell; tested with the stock macOS make 3.81
# and GNU make on Linux.  Use a Python that the package supports (3.10-3.12):
#
#     make check PYTHON=python3.12

PYTHON ?= python3
PIP    := $(PYTHON) -m pip
# Invoked through the selected interpreter so every target uses the same
# environment; `wheel-check` and `wheel-gate` instead use the `eiger2` console
# script of the temporary environment they install the built wheel into.
EIGER2 ?= $(PYTHON) -m eiger2.cli

# The software scope already validated in CI.
LINT_PATHS := Python/eiger2 tests tools examples
TEST_PATHS := tests

DIST_DIR ?= dist
# One source of the version: Python/eiger2/__init__.py.
VERSION  := $(shell sed -n 's/^__version__ = "\(.*\)"$$/\1/p' $(CURDIR)/Python/eiger2/__init__.py)

# Helper scripts.  Both are ordinary Python scripts with their own tests in
# `tests/`, so a target's verdict is tested logic rather than shell.
#
# `check_smoke_manifest.py` is stdlib-only, and so are `check_wheel.py`'s
# `environment`, `contents` and `sdist` modes: they inspect PYTHONPATH and read
# the archives without installing or importing them.  `check_wheel.py
# installed` is deliberately not stdlib-only -- after auditing provenance it
# imports the installed eiger2 and runs one small permutation-FDR call, so that
# the artefact a user installs is exercised rather than only opened.  Those
# imports resolve inside the isolated environment the target just built the
# wheel into, never from the checkout; the mode's own path checks enforce that.
WHEEL_CHECKER   := $(CURDIR)/tools/check_wheel.py
MANIFEST_CHECKER := $(CURDIR)/tools/check_smoke_manifest.py
EXAMPLE_SCRIPT  := $(CURDIR)/examples/make_example_data.py

# Optional modules a default install must not contain.
DEFAULT_ABSENT := --require-absent numba --require-absent sklearn \
                  --require-absent diptest

.DEFAULT_GOAL := help
.PHONY: help install-dev lint test build wheel-check wheel-gate \
        smoke-newer smoke-legacy smoke check

help: ## Show this help
	@echo "EIGER $(VERSION) -- software development tasks"
	@echo
	@awk 'BEGIN {FS = ":.*## "} /^[a-z][a-z-]*:.*## / \
	      {printf "  make %-13s %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo
	@echo "  PYTHON=$(PYTHON)  (override to select an interpreter)"
	@echo "  Smoke and wheel targets work in temporary directories only."

install-dev: ## Install the package editably with the legacy and dev extras
	$(PIP) install -e ".[legacy,dev]"

lint: ## Ruff over the package, tests, tools and example
	$(PYTHON) -m ruff check $(LINT_PATHS)

test: ## Run the software test suite
	$(PYTHON) -m pytest $(TEST_PATHS) -q

build: ## Build the sdist and wheel into dist/ (dist/ is never cleaned)
	@echo "[build] writing to $(DIST_DIR)/ -- the directory is never cleaned;"
	@echo "[build] an artefact of the same name is replaced by the new build"
	$(PYTHON) -m build --outdir $(DIST_DIR) .

# ---------------------------------------------------------------------------
# Shared building blocks
#
# `wheel-check`, `wheel-gate` and the smoke targets are assembled from these so
# there is one implementation of each step rather than one per target, and so
# CI can run exactly what a contributor runs.  Each block is one `&&` chain in
# the calling recipe's single shell: `TMP`, `WHEEL` and the `SMOKE_*` variables
# set by an earlier block are visible to a later one.  `$(1)` is the label the
# calling target prints under.
# ---------------------------------------------------------------------------

# One temporary directory per invocation, outside the repository, removed on
# exit.  The trap removes only the directory mktemp just created.
define tmp_setup
TMP="$$(mktemp -d "$${TMPDIR:-/tmp}/eiger-$(1).XXXXXX")" && \
trap 'test -n "$$TMP" && rm -rf "$$TMP"' EXIT HUP INT TERM && \
echo "[$(1)] temporary directory: $$TMP"
endef

# The forbidden directories every wheel target checks against: the checkout,
# and under Actions the workspace as well.
FORBID := --forbid "$(CURDIR)" $${GITHUB_WORKSPACE:+--forbid "$$GITHUB_WORKSPACE"}

# Isolated sdist and wheel into $$TMP/dist rather than dist/, so a
# contributor's own distributions are never overwritten.  `python -m build`
# builds the sdist first and then the wheel FROM that sdist, so a wheel that
# passes the audit also shows the sdist is self-contained.  Exactly one of each
# must exist; `check_wheel.py contents` and `sdist` then audit the archives --
# filename, version, platform tag, modules, requirements, licence, entry point.
#
# PYTHONPATH is inspected before anything is built: a checkout on it does not
# merely shadow an import, it makes pip read the editable install's
# `eiger.egg-info` and skip installing the wheel at all.
define wheel_build
$(PYTHON) "$(WHEEL_CHECKER)" --version $(VERSION) environment $(FORBID) && \
{ $(PYTHON) -m build --outdir "$$TMP/dist" "$(CURDIR)" > "$$TMP/build.log" 2>&1 \
  || { cat "$$TMP/build.log" >&2; exit 1; } ; } && \
echo "[$(1)] isolated sdist and wheel built (wheel built from the sdist)" && \
set -- "$$TMP"/dist/*.whl && \
if [ "$$#" -ne 1 ]; then \
    echo "[$(1)] expected exactly one wheel, found $$#" >&2; exit 1; \
fi && \
WHEEL="$$1" && \
test -f "$$WHEEL" && \
set -- "$$TMP"/dist/*.tar.gz && \
if [ "$$#" -ne 1 ]; then \
    echo "[$(1)] expected exactly one sdist, found $$#" >&2; exit 1; \
fi && \
SDIST="$$1" && \
echo "[$(1)] candidate wheel: $$(basename "$$WHEEL"); sdist: $$(basename "$$SDIST")" && \
$(PYTHON) "$(WHEEL_CHECKER)" --version $(VERSION) contents "$$WHEEL" && \
$(PYTHON) "$(WHEEL_CHECKER)" --version $(VERSION) sdist "$$SDIST"
endef

# A fresh virtual environment outside the repository, installed from the built
# wheel WITHOUT extras and never from `.`, an index copy of eiger or an
# editable checkout.  pip's own output names the wheel it processed, and
# check_wheel.py later confirms the same provenance from installed metadata.
define wheel_venv
$(PYTHON) -m venv "$$TMP/venv" && \
"$$TMP/venv/bin/python" -m pip install --disable-pip-version-check "$$WHEEL" && \
"$$TMP/venv/bin/python" -m pip check && \
echo "[$(1)] default install (no extras): pip check ok"
endef

# Every import and CLI check runs from $$TMP/run, which is not the source
# checkout, with the checker copied there so even its own directory on
# sys.path is a temporary one.  PYTHONPATH is deliberately not unset: the
# checker inspects it and fails if it -- or sys.path, or the working
# directory, or the imported module itself -- reaches into the checkout or,
# under Actions, into $$GITHUB_WORKSPACE.  `$(2)` is extra `installed` options.
define wheel_imports
mkdir -p "$$TMP/run" && \
cp "$(WHEEL_CHECKER)" "$$TMP/run/check_wheel.py" && \
( cd "$$TMP/run" && "$$TMP/venv/bin/python" check_wheel.py \
      --version $(VERSION) installed --wheel "$$WHEEL" $(FORBID) $(2) ) && \
( cd "$$TMP/run" && "$$TMP/venv/bin/eiger2" --help > /dev/null ) && \
test "$$(cd "$$TMP/run" && "$$TMP/venv/bin/eiger2" --version)" = "eiger $(VERSION)" && \
echo "[$(1)] eiger2 console entry point ok (eiger $(VERSION))"
endef

# The legacy extra added to the same environment from the same wheel.
define wheel_legacy_extra
"$$TMP/venv/bin/python" -m pip install --disable-pip-version-check "$$WHEEL[legacy]" && \
"$$TMP/venv/bin/python" -m pip check && \
echo "[$(1)] legacy extra installed: pip check ok"
endef

# ---------------------------------------------------------------------------
# Installed-wheel checks
# ---------------------------------------------------------------------------

wheel-check: ## Build sdist and wheel in isolation; audit both; check a default install
	@$(call tmp_setup,wheel-check) && \
	$(call wheel_build,wheel-check) && \
	$(call wheel_venv,wheel-check) && \
	$(call wheel_imports,wheel-check,$(DEFAULT_ABSENT))

# The complete installed-wheel gate CI runs: one build, one temporary
# environment, every documented command driven through the wheel's own console
# script -- first as a default install (the legacy option must then fail with
# the install hint), then with the legacy extra added.
wheel-gate: ## The full installed-wheel gate: build, audit, default and legacy installs, smoke runs
	@$(call tmp_setup,wheel-gate) && \
	$(call wheel_build,wheel-gate) && \
	$(call wheel_venv,wheel-gate) && \
	$(call wheel_imports,wheel-gate,$(DEFAULT_ABSENT)) && \
	SMOKE_PY="$$TMP/venv/bin/python" && \
	SMOKE_EIGER2="$$TMP/venv/bin/eiger2" && \
	echo "[wheel-gate] smoke runs use the installed wheel: $$SMOKE_EIGER2" && \
	cd "$$TMP/run" && \
	$(smoke_inputs) && \
	$(smoke_newer) && \
	$(smoke_matrix) && \
	$(smoke_legacy_needs_extra) && \
	$(call wheel_legacy_extra,wheel-gate) && \
	$(call wheel_imports,wheel-gate,--legacy) && \
	$(smoke_legacy) && \
	$(smoke_compare) && \
	echo "[wheel-gate] installed wheel audited; every documented command passed"

# ---------------------------------------------------------------------------
# Smoke runs
#
# The example generator is copied into a private temporary directory and
# writes there, so the checkout is never touched.  `SMOKE_PY` and
# `SMOKE_EIGER2` are the interpreter and command under test: the selected
# interpreter for `make smoke`, the temporary wheel environment's own
# executables for `make wheel-gate`.  Every run's outputs are judged by
# tools/check_smoke_manifest.py from the manifest and the tables it describes.
# ---------------------------------------------------------------------------
define smoke_inputs
cp "$(EXAMPLE_SCRIPT)" "$$TMP/make_example_data.py" && \
( cd "$$TMP" && $$SMOKE_PY make_example_data.py --out "$$TMP/data" > "$$TMP/generate.log" ) && \
echo "[smoke] synthetic example generated" && \
DATA="$$TMP/data"
endef

define smoke_check
$$SMOKE_PY "$(MANIFEST_CHECKER)" "$$OUT" --exit-code "$$rc" --version $(VERSION)
endef

define smoke_newer
OUT="$$TMP/out_newer" && \
echo "[smoke] eiger2 call (default profile)" && \
{ $$SMOKE_EIGER2 call --betas "$$DATA/betas.tsv" --locations "$$DATA/cpgs.tsv" \
      --out "$$OUT" --quiet; rc=$$?; } && \
$(smoke_check) --profile newer --axis-mode descriptive
endef

define smoke_matrix
OUT="$$TMP/out_matrix" && \
echo "[smoke] eiger2 call --matrix (single-table input)" && \
{ $$SMOKE_EIGER2 call --matrix "$$DATA/matrix.tsv" --out "$$OUT" --quiet; rc=$$?; } && \
$(smoke_check) --profile newer && \
cmp -s "$$OUT/regions.csv" "$$TMP/out_newer/regions.csv" && \
echo "[smoke] --matrix regions.csv identical to --betas/--locations"
endef

define smoke_legacy_needs_extra
echo "[smoke] eiger2 call --legacy without the legacy extra must refuse" && \
{ $$SMOKE_EIGER2 call --betas "$$DATA/betas.tsv" --locations "$$DATA/cpgs.tsv" \
      --out "$$TMP/out_legacy_no_extra" --legacy --quiet 2> "$$TMP/no_extra.err"; \
  rc=$$?; } && \
if [ "$$rc" -ne 1 ] || ! grep -q "pip install numba scikit-learn" "$$TMP/no_extra.err"; then \
    echo "[smoke] expected exit 1 with the install hint, got $$rc:" >&2; \
    cat "$$TMP/no_extra.err" >&2; exit 1; \
fi && \
echo "[smoke] refused with the install hint, as documented"
endef

define smoke_legacy
OUT="$$TMP/out_legacy" && \
echo "[smoke] eiger2 call --legacy" && \
{ $$SMOKE_EIGER2 call --betas "$$DATA/betas.tsv" --locations "$$DATA/cpgs.tsv" \
      --out "$$OUT" --legacy --quiet; rc=$$?; } && \
$(smoke_check) --profile legacy
endef

define smoke_compare
echo "[smoke] eiger2 compare" && \
$$SMOKE_EIGER2 compare --regions-a "$$TMP/out_newer/regions.csv" \
    --regions-b "$$TMP/out_legacy/regions.csv" --out "$$TMP/out_compare" > /dev/null && \
test -s "$$TMP/out_compare/comparison_summary.csv" && \
echo "[smoke] comparison_summary.csv written"
endef

define smoke_under_test
SMOKE_PY="$(PYTHON)" && SMOKE_EIGER2="$(EIGER2)"
endef

smoke-newer: ## Default caller over a temporary copy of the synthetic example
	@$(call tmp_setup,smoke) && \
	$(smoke_under_test) && \
	$(smoke_inputs) && \
	$(smoke_newer)

smoke-legacy: ## Legacy caller over a temporary copy of the synthetic example
	@$(call tmp_setup,smoke) && \
	$(smoke_under_test) && \
	$(smoke_inputs) && \
	$(smoke_legacy)

smoke: ## Every documented command over one generated temporary input
	@$(call tmp_setup,smoke) && \
	$(smoke_under_test) && \
	$(smoke_inputs) && \
	$(smoke_newer) && \
	$(smoke_matrix) && \
	$(smoke_legacy) && \
	$(smoke_compare)

check: lint test wheel-check smoke ## The complete local software gate
	@echo "[check] lint, software tests, isolated wheel check and smoke runs passed"
