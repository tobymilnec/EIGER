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
# environment; `wheel-check` exercises the installed `eiger2` console script.
EIGER2 ?= $(PYTHON) -m eiger2.cli

# The software scope already validated in CI.
LINT_PATHS := Python/eiger2 tests
TEST_PATHS := tests

DIST_DIR ?= dist
VERSION  := 2.2.0

.DEFAULT_GOAL := help
.PHONY: help install-dev lint test build wheel-check \
        smoke-revised smoke-legacy smoke check

help: ## Show this help
	@echo "EIGER $(VERSION) -- software development tasks"
	@echo
	@awk 'BEGIN {FS = ":.*## "} /^[a-z][a-z-]*:.*## / \
	      {printf "  make %-13s %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo
	@echo "  PYTHON=$(PYTHON)  (override to select an interpreter)"
	@echo "  Smoke and wheel-check work in temporary directories only."

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
# Installed-wheel check
#
# Builds into a temporary directory rather than dist/ so a contributor's own
# distributions are never overwritten, installs exactly one wheel into a fresh
# virtual environment outside the repository, and runs every import from a
# working directory that is not the source checkout.  PYTHONPATH is never set.
# ---------------------------------------------------------------------------
wheel-check: ## Build a wheel in isolation and validate the installed package
	@TMP="$$(mktemp -d "$${TMPDIR:-/tmp}/eiger-wheelcheck.XXXXXX")" && \
	trap 'test -n "$$TMP" && rm -rf "$$TMP"' EXIT HUP INT TERM && \
	echo "[wheel-check] temporary directory: $$TMP" && \
	{ $(PYTHON) -m build --outdir "$$TMP/dist" . > "$$TMP/build.log" 2>&1 \
	  || { cat "$$TMP/build.log" >&2; exit 1; } ; } && \
	echo "[wheel-check] isolated sdist and wheel built" && \
	set -- "$$TMP"/dist/*.whl && \
	if [ "$$#" -ne 1 ]; then \
	    echo "[wheel-check] expected exactly one wheel, found $$#" >&2; exit 1; \
	fi && \
	WHEEL="$$1" && \
	test -f "$$WHEEL" && \
	echo "[wheel-check] candidate wheel: $$(basename "$$WHEEL")" && \
	$(PYTHON) -m venv "$$TMP/venv" && \
	"$$TMP/venv/bin/python" -m pip install --quiet --disable-pip-version-check "$$WHEEL" && \
	"$$TMP/venv/bin/python" -m pip check && \
	echo "[wheel-check] pip check ok" && \
	mkdir -p "$$TMP/run" && \
	( cd "$$TMP/run" && REPO="$(CURDIR)" "$$TMP/venv/bin/python" -c 'import os, pathlib, sysconfig, eiger2, eiger2.benchmark; p = pathlib.Path(eiger2.__file__).resolve(); sp = pathlib.Path(sysconfig.get_paths()["purelib"]).resolve(); assert sp in p.parents, "eiger2 imported from %s, not the installed environment" % p; assert not str(p).startswith(os.environ["REPO"] + os.sep), "eiger2 imported from the source checkout: %s" % p; assert eiger2.__version__ == "$(VERSION)", "version %s" % eiger2.__version__; print("[wheel-check] eiger2", eiger2.__version__, "imported from", p)' ) && \
	( cd "$$TMP/run" && "$$TMP/venv/bin/eiger2" --help > /dev/null ) && \
	echo "[wheel-check] eiger2 console entry point ok" && \
	( cd "$$TMP/run" && "$$TMP/venv/bin/python" -c 'import functions.load_functions, functions.scoring; print("[wheel-check] legacy functions package imports")' ) && \
	$(PYTHON) -c 'import sys, zipfile; names = sorted({n.split("/")[0] for n in zipfile.ZipFile(sys.argv[1]).namelist()}); extra = [n for n in names if n not in ("eiger2", "functions") and not n.endswith(".dist-info")]; sys.exit("[wheel-check] unexpected wheel contents: %s" % extra) if extra else print("[wheel-check] wheel top level:", ", ".join(names))' "$$WHEEL"

# ---------------------------------------------------------------------------
# Smoke calls
#
# `make_example_data.py` writes into `data/` beside its own file, so copying it
# into a private temporary directory leaves examples/data and examples/out_*
# untouched.  The generator's own behaviour is unchanged.  The trap removes
# only the directory mktemp just created.
# ---------------------------------------------------------------------------
define smoke_setup
TMP="$$(mktemp -d "$${TMPDIR:-/tmp}/eiger-smoke.XXXXXX")" && \
trap 'test -n "$$TMP" && rm -rf "$$TMP"' EXIT HUP INT TERM && \
echo "[smoke] temporary directory: $$TMP" && \
cp examples/make_example_data.py "$$TMP/make_example_data.py" && \
( cd "$$TMP" && $(PYTHON) make_example_data.py > "$$TMP/generate.log" ) && \
echo "[smoke] synthetic example generated" && \
BETAS="$$TMP/data/betas.tsv" && LOCATIONS="$$TMP/data/cpgs.bed"
endef

# Exit 0 succeeds; exit 1 is the documented refusal state and is passed to the
# manifest check rather than judged here, so the reason is read off the
# manifest and printed before the target fails; anything above 1 is a command
# or data failure and fails immediately.  tools/check_smoke_manifest.py then
# decides on the manifest's own evidence -- see its module docstring.
define smoke_revised
OUT="$$TMP/out_revised" && \
echo "[smoke] revised caller" && \
{ $(EIGER2) call --betas "$$BETAS" --locations "$$LOCATIONS" --out "$$OUT"; rc=$$?; } && \
if [ "$$rc" -gt 1 ]; then \
    echo "[smoke] revised caller failed with exit $$rc" >&2; exit 1; \
elif [ "$$rc" -eq 1 ]; then \
    echo "[smoke] revised caller exited 1 -- inspecting the manifest before judging the run"; \
fi && \
test -s "$$OUT/regions.csv" && \
test -f "$$OUT/run_manifest.json" && \
$(PYTHON) tools/check_smoke_manifest.py "$$OUT/run_manifest.json" --exit-code "$$rc"
endef

define smoke_legacy
OUT="$$TMP/out_legacy" && \
echo "[smoke] legacy caller (exit 0 required)" && \
$(EIGER2) call --betas "$$BETAS" --locations "$$LOCATIONS" --out "$$OUT" --legacy && \
test -s "$$OUT/regions.csv" && \
echo "[smoke] legacy caller exit 0; regions.csv non-empty"
endef

smoke-revised: ## Revised caller over a temporary copy of the synthetic example
	@$(smoke_setup) && \
	$(smoke_revised)

smoke-legacy: ## Legacy caller over a temporary copy of the synthetic example
	@$(smoke_setup) && \
	$(smoke_legacy)

smoke: ## Both smoke calls, sharing one generated temporary input
	@$(smoke_setup) && \
	$(smoke_revised) && \
	$(smoke_legacy)

check: lint test wheel-check smoke ## The complete local software gate
	@echo "[check] lint, software tests, isolated wheel check and both smoke calls passed"
