"""Spec §2a.6: `from eiger2 import *` was broken because `__all__` contained the
literal string `"# probability layer"` -- a comment mistakenly written into the
list. Every name in `__all__` must actually resolve on the package (and on
`eiger2.core`, which has its own `__all__` for `from eiger2.core import *`).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Python"))


def test_star_import_succeeds_and_every_name_in_all_resolves():
    namespace: dict = {}
    exec("from eiger2 import *", namespace)

    import eiger2

    for name in eiger2.__all__:
        assert name in namespace, f"{name!r} in eiger2.__all__ did not import"
        assert getattr(eiger2, name) is namespace[name]


def test_core_star_import_succeeds_and_every_name_in_all_resolves():
    namespace: dict = {}
    exec("from eiger2.core import *", namespace)

    import eiger2.core as core

    for name in core.__all__:
        assert name in namespace, f"{name!r} in eiger2.core.__all__ did not import"
        assert getattr(core, name) is namespace[name]
