"""Core configuration and request-time dependencies.

THE `dependencies` IMPORT HERE IS LAZY, AND THAT IS DELIBERATE.

This package previously did::

    from .config import Settings, get_settings
    from .dependencies import Principal, require_admin, ...

which created a circular import:

    core/__init__  ->  core.dependencies  ->  db.session
                   ->  core.config        ->  core/__init__   (partial)

It only ever worked because `imatch_api.main` imports modules in an order that
happens to resolve the cycle before anything needs it. Importing any submodule
directly -- which is what a test, a script or a CLI tool does -- failed with
"cannot import name 'get_session' from partially initialized module".

`config` is safe to import eagerly: it has no intra-package dependencies.
`dependencies` is not, because it pulls in `db.session`, which needs
`core.config` and therefore re-enters this module.

So `dependencies` is resolved on first attribute access via module __getattr__
(PEP 562). By then every module involved is fully initialised and the cycle
cannot occur. `from imatch_api.core import Principal` still works exactly as
before; it is simply resolved later.

Do not "tidy" these back into eager imports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

# Safe eagerly: config has no intra-package dependencies.
from .config import Settings, get_settings

if TYPE_CHECKING:  # for type checkers and IDEs only; not executed at runtime
    from .dependencies import (
        Principal,
        require_admin,
        require_investigator,
        require_supervisor,
    )

_LAZY = {
    "Principal",
    "require_admin",
    "require_investigator",
    "require_supervisor",
}


def __getattr__(name: str) -> Any:
    """Resolve dependency symbols on first access, breaking the import cycle."""
    if name in _LAZY:
        from . import dependencies

        return getattr(dependencies, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _LAZY)


__all__ = [
    "Principal",
    "Settings",
    "get_settings",
    "require_admin",
    "require_investigator",
    "require_supervisor",
]
