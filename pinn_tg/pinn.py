"""PINN for the Taylor-Gortler (secondary-flow) vortex field in a
streamwise-rotating channel, Re_tau=180, Ro_tau=30.

Unknown: stream function phi(y, z) of the secondary-flow deviation
(z-mean exactly zero by construction).  Spectral ansatz in z:

    phi(y,z) = (1 - y^2)^2 * sum_k [ a_k(y) sin(k z*) + b_k(y) cos(k z*) ],
    z* = 2 pi z / lambda_3,

with the y-profiles a_k, b_k learned by an MLP.  The factor (1-y^2)^2
enforces the no-slip walls exactly (phi = d(phi)/dy = 0 at y = +-1  ->
V = W_dev = 0), and the k >= 1-only spectrum makes the z-mean vanish
identically, so the DNS mean profile is reproduced exactly:

    V(y,z)  =  d(phi)/dz
    W(y,z)  =  Wbar(y) - d(phi)/dy          Wbar = DNS W+
    Omega_x =  dWbar/dy - laplacian(phi)

PDE: fluctuation streamwise-vorticity equation (Bradshaw 1987 Eq. 1),
minus its z-mean.  The RHS vortex-stretching term
    (Omega . grad) U = Ox dU/dx + Oy dU/dy + Oz dU/dz
reduces, per the DNS data (only U(y) is non-uniform: dU/dx = dU/dz = 0),
to the single term Omega_y dU/dy; in the present ansatz
Omega_y = dU/dz - dW/dx = 0 (no streamwise modulation u~(y,z) modeled,
and d/dx = 0), so the stretching terms vanish identically.  If a
streamwise modulation u~(y,z) is added in a later phase, the term
(du~/dz)(dU/dy) must be included in the residual.

    [V Om_y + W Om_z] - <V Om_y + W Om_z>_z
        = (dyy - dzz)( -<vw>_vtx ) + dyz( <v2 - w2>_vtx ) + nu lap^2(omega)

Eddy-viscosity closure of the vortex self-stress, nu_t(y) from DNS:
    -<vw>_vtx    =  nu_t ( phi_zz - phi_yy )
    <v2-w2>_vtx  = -4 nu_t phi_yz      (deviatoric Boussinesq part)

At phi = 0 all closure stresses vanish and the fluctuation equation is
satisfied identically; the vortex amplitude is therefore set by the
energy-calibration loss (documented modeling assumption, plan R1).

Numerics: the spectral ansatz lets ALL z-derivatives and wall-factor
derivatives be evaluated analytically; only the y-derivatives of the MLP
output (2m functions of y) use autograd.  The computational graph stays
tiny, which is orders of magnitude faster than nested double-backward
through the assembled field.
"""
from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn as nn

from .structure import (describe, head_rank, is_legacy_equivalent,
                        legacy_structure, normalize_structure, structure_hash)
from .wall import make_wall


def d(f, x):
    """Pointwise derivative df/dx (valid: each f_k depends only on x_k)."""
    return torch.autograd.grad(f, x, grad_outputs=torch.ones_like(f),
                               create_graph=True)[0]


class _Trunk(nn.Module):
    """Hidden MLP of the structured (non-legacy) variants."""

    def __init__(self, width, depth, activation, skip, norm):
        super().__init__()
        self.inp = nn.Linear(1, width)
        self.blocks = nn.ModuleList(
            [nn.Linear(width, width) for _ in range(max(0, depth - 1))])
        self.norms = (nn.ModuleList([nn.LayerNorm(width)
                                     for _ in range(max(0, depth - 1))])
                      if norm == "layernorm" else None)
        self.skip = skip
        self._act_name = activation

    def _act(self, x):
        if self._act_name == "tanh":
            return torch.tanh(x)
        if self._act_name == "sin":
            return torch.sin(x)
        if self._act_name == "gelu":
            return torch.nn.functional.gelu(x)
        raise ValueError(self._act_name)

    def forward(self, y):
        h = self._act(self.inp(y))
        for i, lin in enumerate(self.blocks):
            r = lin(h)
            if self.norms is not None:
                r = self.norms[i](r)
            r = self._act(r)
            h = h + r if self.skip == "residual" else r
        return h


class TGNet(nn.Module):
    """Spectral-in-z stream-function network with hard BCs.

    Structure is declarative: pass ``structure=<dict>`` (see
    :mod:`pinn_tg.structure`).  Omitting it reproduces the legacy network
    *exactly* (module name ``mlp``, identical layer order and init), so all
    pre-v2 checkpoints load with ``strict=True`` and behave bit-identically.
    """

    def __init__(self, lam=None, m=8, width=64, depth=3, out_scale=0.05,
                 structure: dict | None = None):
        super().__init__()
        if structure is None:
            if lam is None:
                raise TypeError("TGNet requires either lam or structure")
            structure = legacy_structure(lam, m, width, depth, out_scale)
        else:
            structure = normalize_structure(structure)
        self._structure = structure
        self.lam = float(structure["lam"])
        self.m = int(structure["m"])
        k = (2.0 * np.pi / self.lam) * torch.arange(1, self.m + 1,
                                                   dtype=torch.float32)
        self.register_buffer("k", k)

        self._wall = make_wall(structure["wall"]["type"],
                               structure["wall"].get("params"))
        self._legacy = is_legacy_equivalent(structure)
        if self._legacy:
            self._build_legacy(structure)
        else:
            self._build_structured(structure)

        active = structure["mask"]["active_modes"]
        self._mask_all = (active == list(range(1, self.m + 1)))
        mask = torch.zeros(1, 2 * self.m)
        for i in active:
            mask[0, i - 1] = 1.0
            mask[0, self.m + i - 1] = 1.0
        # non-persistent: keeps state_dict identical for the default structure
        self.register_buffer("mode_mask", mask, persistent=False)

        fz = structure["freeze"]
        if fz.get("trunk"):
            self._freeze(self._trunk_modules())
        if fz.get("heads"):
            self._freeze(self._head_modules())

    # -- construction ---------------------------------------------------
    def _build_legacy(self, s: dict) -> None:
        """Verbatim copy of the pre-v2 construction (do not reorder!)."""
        width, depth = s["trunk"]["width"], s["trunk"]["depth"]
        out_scale = s["heads"]["out_scale"]
        dims = [1] + [width] * depth + [2 * self.m]
        layers = []
        for i in range(len(dims) - 1):
            lin = nn.Linear(dims[i], dims[i + 1])
            nn.init.xavier_normal_(lin.weight)
            nn.init.zeros_(lin.bias)
            if i == len(dims) - 2:          # small-amplitude start
                lin.weight.data *= out_scale
            layers.append(lin)
            if i < len(dims) - 2:
                layers.append(nn.Tanh())
        self.mlp = nn.Sequential(*layers)

    def _build_structured(self, s: dict) -> None:
        t, h = s["trunk"], s["heads"]
        width = t["width"]
        out_scale = h["out_scale"]
        self.trunk = _Trunk(width, t["depth"], t["activation"], t["skip"],
                            t["norm"])
        modes = h["groups"]
        mode = h["mode"]
        if mode == "shared":
            self.heads = nn.Linear(width, 2 * self.m)
            finals = [self.heads]
        elif mode == "tied_grouped":
            r = head_rank(s)
            self.basis = nn.Linear(width, r, bias=False)
            self.heads = nn.ModuleList(
                [nn.Linear(r, 2 * len(g)) for g in modes])
            finals = list(self.heads) + [self.basis]
        else:  # grouped / per_mode
            self.heads = nn.ModuleList(
                [nn.Linear(width, 2 * len(g)) for g in modes])
            finals = list(self.heads)
        # groups are stored 1-based in the schema; index from 0 here
        self._groups = [[int(i) - 1 for i in g] for g in modes]

        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.xavier_normal_(mod.weight)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)
        for fin in finals:
            fin.weight.data *= out_scale

    @staticmethod
    def _freeze(mods) -> None:
        for mod in mods:
            for p in mod.parameters():
                p.requires_grad_(False)

    def _trunk_modules(self):
        return [self.mlp] if self._legacy else [self.trunk]

    def _head_modules(self):
        if self._legacy:
            return [self.mlp[-1]]
        out = [self.heads]
        if hasattr(self, "basis"):
            out.append(self.basis)
        return out

    # -- evaluation -----------------------------------------------------
    def ab(self, y):
        """Return ``(N, 2m)``: first ``m`` columns are ``a_k``, then ``b_k``."""
        if self._legacy:
            ab = self.mlp(y)
        elif isinstance(self.heads, nn.Linear):
            ab = self.heads(self.trunk(y))
        else:
            h = self.trunk(y)
            if hasattr(self, "basis"):
                h = self.basis(h)
            ab = h.new_zeros((h.shape[0], 2 * self.m))
            for g, head in zip(self._groups, self.heads):
                out = head(h)
                ab = ab.index_copy(1, torch.tensor(g, device=ab.device),
                                   out[:, :len(g)])
                ab = ab.index_copy(1, torch.tensor([self.m + i for i in g],
                                                   device=ab.device),
                                   out[:, len(g):])
        if not self._mask_all:
            ab = ab * self.mode_mask
        return ab

    def wall_derivs(self, y):
        """``(f, f', f'', f''', f'''')`` -- hard-BC wall factor derivatives."""
        return self._wall.derivs(y)

    def forward(self, y, z):
        ab = self.ab(y)
        a, b = ab[:, :self.m], ab[:, self.m:]
        kz = self.k * z
        spec = (a * torch.sin(kz) + b * torch.cos(kz)).sum(
            dim=1, keepdim=True)
        return self._wall.value(y) * spec

    # -- introspection ---------------------------------------------------
    @property
    def structure(self) -> dict:
        return json.loads(json.dumps(self._structure))

    @property
    def structure_hash(self) -> str:
        return structure_hash(self._structure)

    @property
    def param_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))

    @property
    def is_legacy(self) -> bool:
        return self._legacy

    def describe(self) -> str:
        return describe(self._structure)



class TGProblem:
    """Collocation grid, DNS profiles, residual and loss evaluation."""

    def __init__(self, profiles, lam=1.5, ny=201, nz=64, beta=0.05,
                 nu=1.0 / 180.0, use_nut=True, use_aniso=True):
        self.p = profiles
        self.lam = float(lam)
        self.beta = float(beta)
        self.nu = float(nu)
        self.use_nut = bool(use_nut)
        self.use_aniso = bool(use_aniso)
        self.ny, self.nz = int(ny), int(nz)
        self.U_s2 = self.beta * profiles["vv_max"]      # amplitude^2 scale
        # mean profiles and analytic y-derivatives
        self.Wf = profiles["W"]
        self.Wp = profiles["W"].deriv(1)
        self.Wpp = profiles["W"].deriv(2)
        self.nutf = profiles["nut"]
        self.nutp = profiles["nut"].deriv(1)
        self.nutpp = profiles["nut"].deriv(2)
        # residual scale: eddy diffusion of the target vortex
        yg = torch.linspace(-1.0, 1.0, 101)
        with torch.no_grad():
            nut_typ = float(profiles["nut"](yg).mean())
        k1 = 2.0 * np.pi / self.lam
        self.R_scale = (self.nu + nut_typ) * k1 ** 2 * np.sqrt(
            max(self.U_s2, 1e-12))
        self.base_grid, self.shape = self.make_grid(self.ny, self.nz)

    def make_grid(self, ny, nz, zmax=None):
        yl = torch.linspace(-1.0, 1.0, ny)
        zmax = self.lam if zmax is None else float(zmax)
        zl = torch.arange(nz, dtype=torch.float32) / nz * zmax
        Y, Z = torch.meshgrid(yl, zl, indexing="ij")
        Yf = Y.reshape(-1, 1).contiguous().requires_grad_(True)
        Zf = Z.reshape(-1, 1).contiguous()
        return (Yf, Zf), (ny, nz)

    # ------------------------------------------------------------------
    def _spectral_parts(self, net, y, z):
        """All spectral sums needed by the residual (semi-analytic).

        Base scalar fields (N,1):
            P0   = sum_k (a_k sin + b_k cos)
            Q0   = sum_k k (a_k cos - b_k sin)
            Pk0  = sum_k k^2 (a_k sin + b_k cos)
            Q30  = sum_k k^3 (a_k cos - b_k sin)
            Pk40 = sum_k k^4 (a_k sin + b_k cos)
        Their y-derivatives (P_j = d^j P0/dy^j etc.) are obtained with
        pointwise autograd, which is valid because each grid point depends
        only on its own (y, z).
        """
        m = net.m
        k = net.k                                    # (m,)
        ab = net.ab(y)                              # (N, 2m)
        a, b = ab[:, :m], ab[:, m:]
        S = torch.sin(k * z)                         # (N, m)
        C = torch.cos(k * z)
        base = a * S + b * C
        quad = a * C - b * S
        P0 = base.sum(1, keepdim=True)
        Q0 = (k * quad).sum(1, keepdim=True)
        Pk0 = (k ** 2 * base).sum(1, keepdim=True)
        Q30 = (k ** 3 * quad).sum(1, keepdim=True)
        Pk40 = (k ** 4 * base).sum(1, keepdim=True)
        P1 = d(P0, y)
        P2 = d(P1, y)
        P3 = d(P2, y)
        P4 = d(P3, y)
        Q1 = d(Q0, y)
        Q2 = d(Q1, y)
        Pk1 = d(Pk0, y)
        Pk2 = d(Pk1, y)
        Q31 = d(Q30, y)
        return dict(P0=P0, P1=P1, P2=P2, P3=P3, P4=P4, Q0=Q0, Q1=Q1, Q2=Q2,
                    Pk0=Pk0, Pk1=Pk1, Pk2=Pk2, Q30=Q30, Q31=Q31, Pk40=Pk40)

    # ------------------------------------------------------------------
    def _fields(self, net, grid, shape):
        y, z = grid
        ny, nz = shape
        sp = self._spectral_parts(net, y, z)
        P0, P1, P2, P3, P4 = (sp["P0"], sp["P1"], sp["P2"], sp["P3"],
                              sp["P4"])
        Pk0, Pk1, Pk2 = sp["Pk0"], sp["Pk1"], sp["Pk2"]
        Q0, Q1, Q2 = sp["Q0"], sp["Q1"], sp["Q2"]
        Q30, Q31 = sp["Q30"], sp["Q31"]

        # wall factor f = (1 - y^2)^2 (or a configured variant) and its
        # y-derivatives; ``power2`` returns the exact legacy expressions.
        f, fp, fpp, fppp, fpppp = net.wall_derivs(y)

        # ---- phi derivatives (semi-analytic) ------------------------
        phi = f * P0
        V = f * Q0                                     # phi_z
        Wy = fp * P0 + f * P1                          # phi_y
        Wyy = fpp * P0 + 2.0 * fp * P1 + f * P2
        Wzz = -f * Pk0
        Wyz = fp * Q0 + f * Q1
        Wyyy = fppp * P0 + 3.0 * fpp * P1 + 3.0 * fp * P2 + f * P3
        Wyzz = -(fp * Q30 + f * Q31)
        Wzzz = -f * Q30
        Wyyzz = -(fpp * Pk0 + 2.0 * fp * Pk1 + f * Pk2)
        Wyyyy = (fpppp * P0 + 4.0 * fppp * P1 + 6.0 * fpp * P2
                 + 4.0 * fp * P3 + f * P4)
        Wzzzz = f * sp["Pk40"]

        # ---- mean profiles and vorticity ------------------------------
        Wbar = self.Wf(y)
        Wp = self.Wp(y)
        Wpp = self.Wpp(y)
        W = Wbar - Wy
        Wt = -Wy
        Om = Wp - Wyy - Wzz
        Om_y = Wpp - Wyyy - Wyzz
        Om_z = -(fpp * Q0 + 2.0 * fp * Q1 + f * Q2) + f * Q30

        # ---- nonlinear advection (minus its z-mean) --------------------
        # RHS vortex stretching (Omega . grad)U is zero here: DNS data has
        # only U(y) non-uniform (dU/dx = dU/dz = 0), so the sole surviving
        # term would be Omega_y * dU/dy, but Omega_y = dU/dz - dW/dx = 0
        # in this ansatz (no u~(y,z) modulation; d/dx = 0).
        A = V * Om_y + W * Om_z
        A2 = A.view(ny, nz)
        R1 = (A2 - A2.mean(dim=1, keepdim=True)).reshape(-1, 1)

        # ---- eddy-viscosity closure terms -------------------------------
        if self.use_nut:
            nut = self.nutf(y)
            nutp = self.nutp(y)
            nutpp = self.nutpp(y)
        else:
            nut = nutp = nutpp = torch.zeros_like(Wbar)

        # S' = phi_zz - phi_yy  (fluctuation shear)
        Sy = f * Pk0 - (fpp * P0 + 2.0 * fp * P1 + f * P2)
        Sy_y = (fp * Pk0 + f * Pk1
                - (fppp * P0 + 3.0 * fpp * P1 + 3.0 * fp * P2 + f * P3))
        Sy_yy = (fpp * Pk0 + 2.0 * fp * Pk1 + f * Pk2
                 - (fpppp * P0 + 4.0 * fppp * P1 + 6.0 * fpp * P2
                    + 4.0 * fp * P3 + f * P4))
        Sy_zz = f * sp["Pk40"] + fpp * Pk0 + 2.0 * fp * Pk1 + f * Pk2
        # (dyy - dzz)( nu_t S' )
        T1 = (nutpp * Sy + 2.0 * nutp * Sy_y + nut * Sy_yy
              - nut * Sy_zz)
        # dyz( -4 nu_t phi_yz )
        if self.use_aniso:
            T2 = -4.0 * (-nutp * (fp * Q30 + f * Q31)
                         + nut * (fpp * Pk0 + 2.0 * fp * Pk1 + f * Pk2))
        else:
            T2 = torch.zeros_like(R1)

        # ---- viscous -----------------------------------------------------
        visc = self.nu * (Wyyyy + 2.0 * Wyyzz + Wzzzz)

        R = R1 - T1 - T2 + visc
        return dict(phi=phi, V=V, W=W, Wt=Wt, Wbar=Wbar, Om=Om, R=R,
                    lap=Wyy + Wzz, nut=nut, y=y, z=z,
                    phi_y=Wy, phi_yy=Wyy, phi_yz=Wyz,
                    R1=R1, T1=T1, T2=T2, visc=visc)

    def fields(self, net, grid=None, shape=None):
        """Evaluate all fields on the given grid (default: training grid)."""
        if grid is None:
            return self._fields(net, self.base_grid, (self.ny, self.nz))
        return self._fields(net, grid, shape)

    def eval_grid(self, net, ny, nz):
        """Evaluate all fields on a new (ny, nz) grid (post-processing)."""
        grid, shape = self.make_grid(ny, nz)
        f = self._fields(net, grid, shape)
        return grid, shape, f

    # ------------------------------------------------------------------
    def losses(self, net, grid=None, shape=None):
        if grid is None:
            f = self._fields(net, self.base_grid, (self.ny, self.nz))
            ny, nz = self.ny, self.nz
            y = self.base_grid[0]
        else:
            f = self._fields(net, grid, shape)
            ny, nz = shape
            y = grid[0]

        R = f["R"]
        # Scale-invariant PDE loss: the fluctuation-vorticity equation is
        # homogeneous in phi (phi = 0 solves it exactly), so a raw ||R||^2
        # loss has its global minimum at the trivial solution and would
        # collapse the vortex.  Normalising by the magnitude of the very
        # terms being balanced removes that incentive: the loss then only
        # rewards a *consistent shape*, while the amplitude is fixed by the
        # energy-calibration loss L_e.
        den = (f["R1"] ** 2 + f["T1"] ** 2 + f["T2"] ** 2
               + f["visc"] ** 2).mean()
        L_pde = (R ** 2).mean() / (den + 1e-30)

        v2 = (f["V"] ** 2).view(ny, nz).mean(dim=1)     # <v~^2>_z(y)
        tgt = (self.beta * self.p["vv"](y.detach())).view(ny, nz)[:, 0]
        # RMS-based calibration: a squared-energy loss has a degenerate
        # saddle at phi = 0 (gradient ~ amplitude), letting the PDE loss
        # collapse the vortex to zero; the RMS difference keeps a nonzero
        # restoring gradient at any amplitude.  clamp_min guards against
        # small negative undershoot of the Chebyshev <vv> fit.
        eps = 1e-6
        L_e = ((v2 + eps).sqrt() - tgt.clamp_min(0.0).sqrt()).pow(2).mean() \
            / self.U_s2

        diag = dict(
            L_pde=float(L_pde.detach()), L_energy=float(L_e.detach()),
            rms_R=float((R ** 2).mean().sqrt().detach()),
            max_V=float(f["V"].abs().max().detach()),
            max_Wt=float(f["Wt"].abs().max().detach()),
            max_Om=float(f["Om"].abs().max().detach()),
        )
        return L_pde, L_e, diag, f
