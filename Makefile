# EIGER software task interface.
#
# Scope: the installed package only -- lint, software tests, isolated build,
# installed-wheel validation and the two example smoke calls.  Paper analyses,
# benchmark harnesses, cohort data and figure targets live in the separate
# `eiger_paper` repository and are deliberately absent here.
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
LINT_PATHS := Python/eiger2 tests
TEST_PATHS := tests

DIST_DIR ?= dist
VERSION  := 2.2.0

# Helper scripts.  Both are ordinary stdlib-only scripts with their own tests
# in `tests/`, so a target's verdict is tested logic rather than shell.
WHEEL_CHECKER   := $(CURDIR)/tools/check_wheel.py
MANIFEST_CHECKER := $(CURDIR)/tools/check_smoke_manifest.py

.DEFAULT_GOAL := help
.PHONY: help install-dev lint test build wheel-check wheel-gate \
        smoke-revised smoke-legacy smoke check

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

lint: ## Ruff over the shipped package and the software tests
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
# contributor's own distributions are never overwritten.  Exactly one wheel
# must exist; `check_wheel.py contents` then audits the archive itself --
# filename, version, platform tag, top-level contents, licence, entry point.
#
# PYTHONPATH is inspected before anything is built: a checkout on it does not
# merely shadow an import, it makes pip read the editable install's
# `eiger.egg-info` and skip installing the wheel at all.
define wheel_build
$(PYTHON) "$(WHEEL_CHECKER)" environment $(FORBID) && \
{ $(PYTHON) -m build --outdir "$$TMP/dist" "$(CURDIR)" > "$$TMP/build.log" 2>&1 \
  || { cat "$$TMP/build.log" >&2; exit 1; } ; } && \
echo "[$(1)] isolated sdist and wheel built" && \
set -- "$$TMP"/dist/*.whl && \
if [ "$$#" -ne 1 ]; then \
    echo "[$(1)] expected exactly one wheel, found $$#" >&2; exit 1; \
fi && \
WHEEL="$$1" && \
test -f "$$WHEEL" && \
echo "[$(1)] candidate wheel: $$(basename "$$WHEEL")" && \
$(PYTHON) "$(WHEEL_CHECKER)" --version $(VERSION) contents "$$WHEEL"
endef

# A fresh virtual environment outside the repository, installed from the built
# wheel and never from `.`, an index or an editable checkout.  `$(1)` is the
# extras suffix (empty, or `[legacy]` when the legacy caller must run); pip's
# own output names the wheel it processed, and check_wheel.py later confirms
# the same provenance from the installed metadata.
define wheel_venv
$(PYTHON) -m venv "$$TMP/venv" && \
"$$TMP/venv/bin/python" -m pip install --disable-pip-version-check "$$WHEEL$(1)" && \
"$$TMP/venv/bin/python" -m pip check && \
echo "[$(2)] pip check ok"
endef

# Every import and CLI check runs from $$TMP/run, which is not the source
# checkout, with the checker copied there so even its own directory on
# sys.path is a temporary one.  PYTHONPATH is deliberately not unset: the
# checker inspects it and fails if it -- or sys.path, or the working
# directory, or the imported module itself -- reaches into the checkout or,
# under Actions, into $$GITHUB_WORKSPACE.
define wheel_imports
mkdir -p "$$TMP/run" && \
cp "$(WHEEL_CHECKER)" "$$TMP/run/check_wheel.py" && \
( cd "$$TMP/run" && "$$TMP/venv/bin/python" check_wheel.py \
      --version $(VERSION) installed --wheel "$$WHEEL" $(FORBID) ) && \
( cd "$$TMP/run" && "$$TMP/venv/bin/eiger2" --help > /dev/null ) && \
echo "[$(1)] eiger2 console entry point ok"
endef

# ---------------------------------------------------------------------------
# Installed-wheel checks
# ---------------------------------------------------------------------------

wheel-check: ## Build a wheel in isolation and validate the installed package
	@$(call tmp_setup,wheel-check) && \
	$(call wheel_build,wheel-check) && \
	$(call wheel_venv,,wheel-check) && \
	$(call wheel_imports,wheel-check)

# The complete installed-wheel gate CI runs: one build, one temporary
# environment, and both smoke calls driven through the wheel's own console
# script.  `wheel-check` is the same gate without the legacy extra or the
# smoke calls, so `make check` stays quick enough to run on every change.
wheel-gate: ## The full installed-wheel gate: build, install, audit, both smoke calls
	@$(call tmp_setup,wheel-gate) && \
	$(call wheel_build,wheel-gate) && \
	$(call wheel_venv,[legacy],wheel-gate) && \
	$(call wheel_imports,wheel-gate) && \
	SMOKE_PY="$$TMP/venv/bin/python" && \
	SMOKE_EIGER2="$$TMP/venv/bin/eiger2" && \
	echo "[wheel-gate] smoke calls run from the installed wheel: $$SMOKE_EIGER2" && \
	cd "$$TMP/run" && \
	$(smoke_inputs) && \
	$(smoke_revised) && \
	$(smoke_legacy) && \
	echo "[wheel-gate] installed wheel audited and both smoke calls passed"

# ---------------------------------------------------------------------------
# Smoke calls
#
# `make_example_data.py` writes into `data/` beside its own file, so copying it
# into a private temporary directory leaves examples/data and examples/out_*
# untouched.  The generator's own behaviour is unchanged.  `SMOKE_PY` and
# `SMOKE_EIGER2` are the interpreter and caller under test: the selected
# interpreter for `make smoke`, the temporary wheel environment's own
# executables for `make wheel-gate`.
# ---------------------------------------------------------------------------
define smoke_inputs
cp "$(CURDIR)/examples/make_example_data.py" "$$TMP/make_example_data.py" && \
( cd "$$TMP" && $$SMOKE_PY make_example_data.py > "$$TMP/generate.log" ) && \
echo "[smoke] synthetic example generated" && \
BETAS="$$TMP/data/betas.tsv" && LOCATIONS="$$TMP/data/cpgs.bed"
endef

# Exit 0 succeeds; exit 1 is the documented `inference_unreliable` refusal and
# is passed to the manifest check rather than judged here, so the reason is
# read off the manifest and printed before the target fails; anything above 1
# is a command or data failure and fails immediately.  The shipped example
# reaches `partial_inference`, which the CLI exits 0 for.
# tools/check_smoke_manifest.py then decides on the manifest's own evidence --
# see its module docstring.
define smoke_revised
OUT="$$TMP/out_revised" && \
echo "[smoke] revised caller" && \
{ $$SMOKE_EIGER2 call --betas "$$BETAS" --locations "$$LOCATIONS" --out "$$OUT"; rc=$$?; } && \
if [ "$$rc" -gt 1 ]; then \
    echo "[smoke] revised caller failed with exit $$rc" >&2; exit 1; \
elif [ "$$rc" -eq 1 ]; then \
    echo "[smoke] revised caller exited 1 -- inspecting the manifest before judging the run"; \
fi && \
test -s "$$OUT/regions.csv" && \
test -f "$$OUT/run_manifest.json" && \
$$SMOKE_PY "$(MANIFEST_CHECKER)" "$$OUT/run_manifest.json" --exit-code "$$rc"
endef

define smoke_legacy
OUT="$$TMP/out_legacy" && \
echo "[smoke] legacy caller (exit 0 required)" && \
$$SMOKE_EIGER2 call --betas "$$BETAS" --locations "$$LOCATIONS" --out "$$OUT" --legacy && \
test -s "$$OUT/regions.csv" && \
echo "[smoke] legacy caller exit 0; regions.csv non-empty"
endef

define smoke_under_test
SMOKE_PY="$(PYTHON)" && SMOKE_EIGER2="$(EIGER2)"
endef

smoke-revised: ## Revised caller over a temporary copy of the synthetic example
	@$(call tmp_setup,smoke) && \
	$(smoke_under_test) && \
	$(smoke_inputs) && \
	$(smoke_revised)

smoke-legacy: ## Legacy caller over a temporary copy of the synthetic example
	@$(call tmp_setup,smoke) && \
	$(smoke_under_test) && \
	$(smoke_inputs) && \
	$(smoke_legacy)

smoke: ## Both smoke calls, sharing one generated temporary input
	@$(call tmp_setup,smoke) && \
	$(smoke_under_test) && \
	$(smoke_inputs) && \
	$(smoke_revised) && \
	$(smoke_legacy)

check: lint test wheel-check smoke ## The complete local software gate
	@echo "[check] lint, software tests, isolated wheel check and both smoke calls passed"
