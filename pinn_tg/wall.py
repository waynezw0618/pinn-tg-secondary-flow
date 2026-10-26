"""Wall-damping factor families for the stream-function ansatz.

All variants satisfy the no-slip/no-penetration conditions exactly::

    f(+-1) = 0,  f'(+-1) = 0

so ``phi = f(y) * S(y, z)`` gives ``v~ = dphi/dz = 0`` and
``w~ = -dphi/dy = 0`` at both walls for any smooth ``S``.

The ``power2`` variant reproduces the pre-v2 closed-form expressions
*verbatim* (same operation order), so the default model stays bit-identical
to the legacy one.

Higher derivatives of the non-polynomial variants are obtained with nested
``autograd.grad`` on the scalar field ``f(y)`` of shape ``(N, 1)`` -- never on
an ``(N, 2m)`` tensor (see the project's autograd convention).
"""
from __future__ import annotations

import torch


def _d(f, x, order: int = 1):
    g = f
    for _ in range(order):
        g = torch.autograd.grad(g, x, grad_outputs=torch.ones_like(g),
                                create_graph=True)[0]
    return g


class WallFactor:
    """Base class: value + y-derivatives up to 4th order."""

    kind = "base"

    def __init__(self, params: dict | None = None):
        self.params = dict(params or {})

    # -- interface -----------------------------------------------------
    def value(self, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def derivs(self, y: torch.Tensor):
        """Return ``(f, f', f'', f''', f'''')`` as ``(N,1)`` tensors."""
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"WallFactor({self.kind}, {self.params})"


class Power2Wall(WallFactor):
    """Classic ``(1 - y^2)^2`` -- identical arithmetic to the legacy code."""

    kind = "power2"

    def value(self, y: torch.Tensor) -> torch.Tensor:
        return (1.0 - y ** 2) ** 2

    def derivs(self, y: torch.Tensor):
        f = (1.0 - y ** 2) ** 2
        fp = -4.0 * y * (1.0 - y ** 2)
        fpp = 4.0 * (3.0 * y ** 2 - 1.0)
        fppp = 24.0 * y
        fpppp = torch.full_like(y, 24.0)
        return f, fp, fpp, fppp, fpppp


class Power2LagWall(WallFactor):
    """``(1-y^2)^2 * (1 + a (1-y^2))`` -- a one-parameter smooth family.

    Still exactly zero with zero slope at both walls; ``a > 0`` thickens the
    near-wall lobe, ``a < 0`` sharpens it.
    """

    kind = "power2_plus_lag"

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.a = float(self.params.get("a", 0.5))

    def value(self, y: torch.Tensor) -> torch.Tensor:
        base = (1.0 - y ** 2) ** 2
        return base * (1.0 + self.a * (1.0 - y ** 2))

    def derivs(self, y: torch.Tensor):
        return _autograd_derivs(self.value, y)


class ExpBlendWall(WallFactor):
    """``(1-y^2)^2 * exp(-a y^2)`` -- asymmetric-free Gaussian-modulated wall.

    The exponential factor does not break the hard BCs because the
    polynomial prefactor carries the double zero at ``y = +-1``.
    """

    kind = "exp_blend"

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.a = float(self.params.get("a", 1.0))

    def value(self, y: torch.Tensor) -> torch.Tensor:
        return (1.0 - y ** 2) ** 2 * torch.exp(-self.a * y ** 2)

    def derivs(self, y: torch.Tensor):
        return _autograd_derivs(self.value, y)


def _autograd_derivs(fn, y: torch.Tensor):
    if not y.requires_grad:
        y = y.clone().requires_grad_(True)
    f = fn(y)
    f1 = _d(f, y, 1)
    f2 = _d(f1, y, 1)
    f3 = _d(f2, y, 1)
    f4 = _d(f3, y, 1)
    return f, f1, f2, f3, f4


_REGISTRY = {
    "power2": Power2Wall,
    "power2_plus_lag": Power2LagWall,
    "exp_blend": ExpBlendWall,
}


def make_wall(kind: str = "power2", params: dict | None = None) -> WallFactor:
    cls = _REGISTRY.get(kind)
    if cls is None:
        raise ValueError(f"unknown wall factor {kind!r}; "
                         f"choose from {sorted(_REGISTRY)}")
    return cls(params)


def wall_param_count() -> int:
    """Wall factors introduce no learnable parameters."""
    return 0
