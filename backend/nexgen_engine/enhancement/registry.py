"""Plugin registry. Adding a model is one class plus one decorator.

Two properties are enforced here rather than left to reviewers:

  * A backend MUST declare its track. There is no default. A contributor adding
    a generative restorer cannot accidentally inherit "measurement" and slip a
    learned prior into the deterministic path.

  * A backend MUST be able to say why it is unavailable. Missing weights on an
    offline host is the normal case, not an error, and the UI needs to say
    "CodeFormer: weights not present" instead of failing the whole request.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .types import Task, Track

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BackendSpec:
    """Static description of a backend. Everything the planner and UI need."""

    name: str
    track: Track
    task: Task
    version: str
    summary: str
    deterministic: bool = True
    requires_weights: bool = False
    requires_torch: bool = False
    # Rough peak VRAM at 512x512, megabytes. Advisory for scheduling only; the
    # runner measures the real figure and reports that instead.
    vram_estimate_mb: float = 0.0
    default_parameters: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "track": self.track.value,
            "task": self.task.value,
            "version": self.version,
            "summary": self.summary,
            "deterministic": self.deterministic,
            "requires_weights": self.requires_weights,
            "requires_torch": self.requires_torch,
            "vram_estimate_mb": self.vram_estimate_mb,
            "default_parameters": dict(self.default_parameters),
        }


class EnhancementBackend(ABC):
    """One model or one classical operator.

    Lifecycle is explicit because the development GPU has 6 GB and cannot hold a
    restorer and the recogniser at once:

        load(device) -> apply(...) [xN] -> release()

    The runner drives it and measures peak VRAM across the window. A backend
    that holds no weights implements load/release as no-ops.
    """

    spec: BackendSpec

    def __init__(self) -> None:
        self._device = "cpu"

    # -- availability ------------------------------------------------------

    def availability(self) -> tuple[bool, str]:
        """``(available, reason_if_not)``. Never raises."""
        return True, ""

    @property
    def available(self) -> bool:
        return self.availability()[0]

    # -- lifecycle ---------------------------------------------------------

    def load(self, device: str = "cpu") -> None:
        self._device = device

    def release(self) -> None:
        """Drop weights and free device memory. Must be idempotent."""

    @property
    def device(self) -> str:
        return self._device

    # -- work --------------------------------------------------------------

    @abstractmethod
    def apply(self, pixels: np.ndarray, parameters: dict[str, Any]) -> np.ndarray:
        """uint8 RGB HWC in, uint8 RGB HWC out. Same contract both ways."""

    def scale_factor(self, parameters: dict[str, Any]) -> float:
        """Output/input linear size. Used by the planner to reason about geometry."""
        return 1.0


_REGISTRY: dict[str, type[EnhancementBackend]] = {}
_INSTANCES: dict[str, EnhancementBackend] = {}


def register(spec: BackendSpec) -> Callable[[type[EnhancementBackend]], type[EnhancementBackend]]:
    """Class decorator. The spec carries the mandatory track declaration."""

    def decorate(cls: type[EnhancementBackend]) -> type[EnhancementBackend]:
        if not isinstance(spec.track, Track):
            raise TypeError(f"{spec.name}: track must be a Track, got {type(spec.track)!r}.")
        if not isinstance(spec.task, Task):
            raise TypeError(f"{spec.name}: task must be a Task, got {type(spec.task)!r}.")
        if spec.name in _REGISTRY:
            raise ValueError(f"Enhancement backend {spec.name!r} is already registered.")
        cls.spec = spec
        _REGISTRY[spec.name] = cls
        return cls

    return decorate


def get_backend(name: str) -> EnhancementBackend:
    """Singleton instance per backend name.

    Instances are cached because constructing one may map weights. They are not
    thread-safe to *apply* concurrently, which is fine: the runner is sequential
    by design on a 6 GB card.
    """
    if name not in _REGISTRY:
        raise KeyError(f"Unknown enhancement backend {name!r}. Registered: {', '.join(sorted(_REGISTRY))}")
    if name not in _INSTANCES:
        _INSTANCES[name] = _REGISTRY[name]()
    return _INSTANCES[name]


def all_specs() -> list[BackendSpec]:
    return [cls.spec for cls in _REGISTRY.values()]


def available_backends(task: Task | None = None, track: Track | None = None) -> list[dict[str, Any]]:
    """Registry contents with live availability, for the UI and the planner."""
    rows: list[dict[str, Any]] = []
    for name in sorted(_REGISTRY):
        spec = _REGISTRY[name].spec
        if task is not None and spec.task is not task:
            continue
        if track is not None and spec.track is not track:
            continue
        try:
            ok, reason = get_backend(name).availability()
        except Exception as exc:  # pragma: no cover - a broken backend must not break the list
            logger.warning("Backend %s failed its availability check: %s", name, exc)
            ok, reason = False, f"availability check failed: {exc}"
        rows.append({**spec.as_dict(), "available": ok, "unavailable_reason": reason})
    return rows


def reset_registry_instances() -> None:
    """Drop cached instances, releasing any held weights. Used by tests."""
    for instance in _INSTANCES.values():
        try:
            instance.release()
        except Exception:  # pragma: no cover - best effort
            pass
    _INSTANCES.clear()


__all__ = [
    "BackendSpec",
    "EnhancementBackend",
    "Task",
    "Track",
    "all_specs",
    "available_backends",
    "get_backend",
    "register",
    "reset_registry_instances",
]
