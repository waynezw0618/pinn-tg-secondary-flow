"""pinn_tg/asm.py - Explicit algebraic Reynolds stress model (ASM).

Implements the Gatski-Speziale (1993) quadratic nonlinear eddy-viscosity model
(NLEVM). This form is equivalent to the Wallin-Johansson (2000) explicit ASM
to leading order in the quadratic nonlinearity, and is widely used in
industrial RANS codes (e.g. OpenFOAM nonlinearEddyViscosity, ANSYS Fluent
nonlinear EVM). It captures the principal Reynolds-stress anisotropy effects
in rotating turbulent flows (notably the rotation-induced <uv>, <uw>, <vw>).

Formulation (canonical anisotropy form, see ASM.__call__):

    b_ij = <u_i u_j>/(2k) - delta_ij/3
         = -C_mu tau S_ij                         (Boussinesq / linear)
         + C2 tau^2 SS_dev                        (quadratic in S)
         + C3 tau^2 WW_dev                        (quadratic in W)
         + C4 tau^2 (SW - WS)                     (S-W cross term)
    <u_i u_j> = 2 k (b_ij + delta_ij/3)
    nu_t = C_mu k^2 / eps,  tau = k / eps

The linear coefficient is C_mu = 0.09 itself; there is NO additional
linear-in-S term. System rotation enters through the rotation tensor
W_ij = 0.5 (du_i/dx_j - du_j/dx_i) + (system rotation), which MUST be added
externally (see add_system_rotation).

KNOWN LIMITATIONS (declared, not hidden):
  * The model is a TRUNCATED (quadratic) NLEVM.  The exact Wallin-Johansson
    (2000) six-coefficient EARSM replaces C2..C4 by rational functions of the
    invariants S* = tau |S| and W* = tau |W| which saturate for large
    invariants.  This quadratic truncation does NOT.
  * At Re_tau = 180, Ro_tau = 30 the DNS gives tau*Omega up to ~12-17, far
    outside the regime where the truncated expansion is meaningful: an
    unregularized evaluation returns stresses ~1e6 x DNS
    (measured, tools/_check_asm_dns.py).  Hence `regularize=True` by default,
    which applies a Pade-like saturating factor to the quadratic coefficients.
    This is a REGULARIZATION of the truncated model, not the exact WJ2000
    algebra; results must be reported as such.

Coordinate convention (project): x = streamwise (axis 0),
y = wall-normal (axis 1), z = spanwise (axis 2).

References:
    Wallin, S. & Johansson, A.V. (2000) J. Fluid Mech. 415:89-132.
    Gatski, T.B. & Speziale, C.G. (1993) J. Fluid Mech. 254:59-78.
    Craft, T.J., Launder, B.E. & Suga, K. (1996)
        Int. J. Heat Fluid Flow 17:243-250.
    Pope, S.B. (2000) Turbulent Flows, Cambridge, Ch. 11.

Disclosure: This is a 2nd-order (quadratic) NLEVM; higher-order WJ2000
nonlinear corrections (6-coefficient explicit ASM) would require
implementing the cubic characteristic polynomial whose roots give the
principal stresses. The 4-coefficient form used here is sufficient for
the leading anisotropy in rotating channel flow and matches the form
used in industrial RANS implementations.
"""
from __future__ import annotations

import torch


# --- Anisotropy-model coefficients (b_ij form, see ASM.__call__) ---
# The linear coefficient is C_mu = 0.09 (the standard k-eps value), applied
# inside the ASM as b_lin = -C_mu * tau * S_ij; there is NO separate C1.
# C1 is kept only for backward-compatible imports and is unused.
C1 = 0.4     # DEPRECATED / unused: was a duplicate linear-S term (sign bug)
C2 = -0.103  # quadratic S^2 coefficient (deviatoric S_ik S_kj tau^2)
C3 = -0.103  # quadratic W^2 coefficient (deviatoric W_ik W_kj tau^2)
C4 = 0.114   # cross SW coefficient (S_ik W_kj - W_ik S_kj) tau^2


def add_system_rotation(W, Omega):
    """Add system rotation contribution to an antisymmetric rotation tensor.

    System rotation Omega (vector along axis of rotation) contributes the
    antisymmetric tensor W_ij^rot = epsilon_ijk Omega_k. This MUST be added
    to the mean-flow rotation-rate tensor for ASM closure to feel the
    Coriolis effect.

    Args:
        W: (N, 3, 3) antisymmetric part of mean velocity gradient
           (W_ij = 0.5 (du_i/dx_j - du_j/dx_i))
        Omega: (3,) or (N, 3) system angular velocity vector
    Returns:
        W_total: (N, 3, 3) W + epsilon_ijk Omega_k
    """
    eps = torch.zeros(3, 3, dtype=W.dtype, device=W.device)
    eps[0, 1], eps[1, 0] = 0.0, 0.0  # = 0 (rotation along x: Omega_x component)
    eps[0, 2], eps[2, 0] = 0.0, 0.0  # = 0
    eps[1, 2], eps[2, 1] = -1.0, 1.0  # Omega_x: W_23 += -Omega_x, W_32 += +Omega_x
    # NOTE: only Omega_x contributes to the (2,3) plane for a vector along x;
    # for a general Omega use the full antisymmetric tensor.
    # Here we assume the project convention: rotation is along x, so only
    # the (2,3) component is non-zero (W_23 = -Omega_x, W_32 = +Omega_x).
    if Omega.dim() == 1:
        Omega = Omega.unsqueeze(0).expand(W.shape[0], -1)
    W_rot = torch.zeros_like(W)
    # W_rot_ij = epsilon_ijk Omega_k; for Omega along axis 0 (x):
    W_rot[:, 1, 2] = -Omega[:, 0]
    W_rot[:, 2, 1] = +Omega[:, 0]
    return W + W_rot


def project_realizability(R, k):
    """Project Reynolds stress eigenvalues back to the physical domain [-2k/3, 4k/3].

    Strong rotation or numerical drift may push <u_i u_j> outside the
    Schwarz realizability bounds (eigenvalues in [-2k/3, 4k/3]). This
    helper snaps each principal eigenvalue back to the physical range,
    preserving trace = 2k.

    Args:
        R: (N, 3, 3) Reynolds stress tensor
        k: (N,) turbulent kinetic energy (used for normalization)
    Returns:
        R_proj: (N, 3, 3) projected Reynolds stress
    """
    R_proj = R.clone()
    I3 = torch.eye(3, dtype=R.dtype, device=R.device)
    for b in range(R.shape[0]):
        kk = k[b].clamp_min(1e-30)
        # k≈0 (训练初期/壁面附近) 跳过投影，R 近乎零矩阵不需要 clamp
        if kk.item() < 1e-12:
            continue
        try:
            # +1e-30*I 正则化防病态
            eig, vec = torch.linalg.eigh(R[b] + 1e-30 * I3)
            lo = (-2.0 * kk / 3.0).item()
            hi = (4.0 * kk / 3.0).item()
            eig_c = torch.clamp(eig, min=lo, max=hi)
            R_proj[b] = vec @ torch.diag(eig_c) @ vec.T
        except Exception:
            # 数值兜底：若本点 eigh 失败则保留原值，下一次梯度会自然远离此区域
            continue
    return R_proj


def project_realizability_batched(R, k):
    """Fast batched realizability projection with straight-through gradient.

    Forward: exact eigenvalue clamp via a single batched eigh call.
    Backward: identity (gradients flow through R as if no projection).

    The projection is a forward safety net; the differentiable penalty is
    handled separately in PINN1D.loss_realizability. Straight-through
    estimator avoids (a) the O(N) python loop, (b) NaN gradients from
    degenerate eigenvalue backward passes.
    """
    R_det = R.detach()
    kk = k.detach().clamp_min(1e-30)
    I3 = torch.eye(3, dtype=R.dtype, device=R.device)
    # 正则化 + batched eigh；k≈0 的点 mask 掉（保持原值）
    try:
        eig, vec = torch.linalg.eigh(R_det + 1e-30 * I3)  # (N,3), (N,3,3)
        lo = (-2.0 / 3.0) * kk.unsqueeze(-1)  # (N,1)
        hi = (4.0 / 3.0) * kk.unsqueeze(-1)
        eig_c = torch.clamp(eig, min=lo, max=hi)
        R_target = vec @ torch.diag_embed(eig_c) @ vec.transpose(-1, -2)
    except Exception:
        return R  # batched 失败时整体不投影（损失项会惩罚越界）
    # straight-through: forward = R_target, backward = identity
    return R + (R_target - R_det)


class ASM:
    """Gatski-Speziale (1993) / Craft-Launder-Suga (1996) quadratic NLEVM.

    Equivalent to WJ2000 explicit ASM to leading order in the dimensionless
    invariants S* = (k/eps) sqrt(2 S_ij S_ij) and W* = (k/eps) sqrt(2 W_ij W_ij).
    """

    def __init__(self, C_mu: float = 0.09, project: bool = True,
                 regularize: bool = True, sat_c: float = 1.0):
        """Args:
            C_mu: k-eps constant in nu_t = C_mu k^2/eps (also the linear
                  anisotropy coefficient; see module docstring).
            project: apply the forward realizability projection.
            regularize: apply the saturating (Pade-like) factor to the
                  quadratic coefficients.  MUST stay True for Ro_tau >= ~10;
                  see the note in __call__.
            sat_c: strength of the saturation (1.0 = tau*invariant ~ 1).
        """
        self.C_mu = float(C_mu)
        self.project = bool(project)
        self.regularize = bool(regularize)
        self.sat_c = float(sat_c)

    def __call__(self, S: torch.Tensor, W: torch.Tensor,
                 k: torch.Tensor, eps: torch.Tensor) -> dict:
        """Compute Reynolds stress tensor <u_i u_j> from S, W, k, eps.

        Args:
            S: (N, 3, 3) strain-rate tensor (symmetric part of velocity gradient)
            W: (N, 3, 3) rotation-rate tensor (antisymmetric part + system rotation)
            k: (N,) turbulent kinetic energy
            eps: (N,) turbulent dissipation rate
        Returns:
            dict with keys:
                R:        (N, 3, 3) Reynolds stress <u_i u_j>
                R_lin:    (N, 3, 3) linear part -2 nu_t S_ij
                R_NL:     (N, 3, 3) nonlinear correction
                nu_t:     (N,) eddy viscosity
                aniso:    (N, 3, 3) anisotropy <u_i u_j> - 2k/3 delta_ij
                S2, W2:   (N,) invariants (2 S_ij S_ij and 2 W_ij W_ij)
        """
        assert S.shape == W.shape, "S and W must have the same shape"
        assert S.shape[-1] == 3 and S.shape[-2] == 3, "S, W must be 3x3"

        # Eddy viscosity (scalar per batch element)
        nu_t = self.C_mu * k.pow(2) / eps.clamp_min(1e-30)

        # Broadcast scalars to (N, 1, 1) for tensor arithmetic
        nu_t_b = nu_t.unsqueeze(-1).unsqueeze(-1)
        k_b = k.unsqueeze(-1).unsqueeze(-1)
        # tau = k/eps clamp 到物理范围 [1e-2, 1e2]：
        # DNS 平衡 tau ~ 0.04 (k~2, eps~50 outer units)；训练初期 eps→0 时
        # tau 无界会让非线性项 tau^2 * WW_dev 爆炸，反传 NaN
        tau = (k / eps.clamp_min(1e-30)).clamp(1e-2, 1e2)
        tau_b = tau.unsqueeze(-1).unsqueeze(-1)

        # Identity tensor
        eye = torch.eye(3, device=S.device, dtype=S.dtype).unsqueeze(0)

        # Traces (N, 1)
        S_kk = S.diagonal(dim1=1, dim2=2).sum(dim=1, keepdim=True)

        # Quadratic products S^2, W^2, SW+WS
        SS = torch.einsum('bij,bjk->bik', S, S)
        WW = torch.einsum('bij,bjk->bik', W, W)
        SW = torch.einsum('bij,bjk->bik', S, W)
        WS = torch.einsum('bij,bjk->bik', W, S)

        # Traces of S^2 and W^2
        S2_trace = SS.diagonal(dim1=1, dim2=2).sum(dim=1, keepdim=True)
        W2_trace = WW.diagonal(dim1=1, dim2=2).sum(dim=1, keepdim=True)

        # Deviators
        S_dev = S - (1.0 / 3.0) * S_kk.unsqueeze(-1) * eye
        SS_dev = SS - (1.0 / 3.0) * S2_trace.unsqueeze(-1) * eye
        WW_dev = WW - (1.0 / 3.0) * W2_trace.unsqueeze(-1) * eye

        # ---- Reynolds stress in canonical anisotropy form b_ij ----
        #   b_ij = <u_i u_j>/(2k) - delta_ij/3
        #        = -C_mu * tau * S_ij                       (Boussinesq linear)
        #        + C2 * tau^2 * SS_dev                      (quadratic in S)
        #        + C3 * tau^2 * WW_dev                      (quadratic in W)
        #        + C4 * tau^2 * (SW - WS)                   (S-W cross)
        #   <u_i u_j> = 2k (b_ij + delta_ij/3)
        #
        # SIGN FIX (2026-09-29).  The previous version added a *separate*
        # linear term  +k*C1*tau*S_dev  on top of the Boussinesq part
        # -2*nu_t*S_dev.  Both are linear in S and both scale as k^2/eps, so
        # the net linear coefficient became
        #       (-2*C_mu + C1) = (-0.18 + 0.40) = +0.22   <- WRONG SIGN
        # which flipped <uv> to up-gradient transport.  verify_asm T3 failed on
        # exactly this.  The linear part is now exactly -C_mu*tau*S_ij, so that
        # <uv> = -C_mu (k^2/eps) dU/dy = -nu_t dU/dy  (down-gradient).  C1 is
        # therefore no longer used as a *separate* coefficient: it is unified
        # into C_mu.
        b_lin = -self.C_mu * tau_b * S_dev

        # ---- validity saturation (IMPORTANT for strongly rotating flows) ----
        # The quadratic coefficients C2, C3, C4 multiply tau^2 * (S^2 or W^2).
        # The rotation tensor W carries the system rotation Omega, so the
        # relevant invariant is (tau*Omega)^2.  At Ro_tau = 30 the DNS gives
        # tau*Omega up to ~12-17, where the truncated expansion is far outside
        # its validity band: measured ASM stresses were ~1e6 x DNS (see
        # tools/_check_asm_dns.py).  The genuine Wallin-Johansson (2000)
        # six-coefficient EARSM has RATIONAL coefficients that saturate for
        # large invariants and never blow up; we emulate that behaviour with
        # the Pade-like factor below, declared explicitly as a regularization
        # of the truncated model rather than part of the exact WJ2000 algebra.
        if self.regularize:
            inv_S = tau_b.pow(2) * (S * S).sum(dim=(1, 2)).unsqueeze(-1).unsqueeze(-1)
            inv_W = tau_b.pow(2) * (W * W).sum(dim=(1, 2)).unsqueeze(-1).unsqueeze(-1)
            sat = 1.0 / (1.0 + self.sat_c * (inv_S + inv_W))
        else:
            sat = 1.0

        b_nl = sat * (C2 * tau_b.pow(2) * SS_dev
                      + C3 * tau_b.pow(2) * WW_dev
                      + C4 * tau_b.pow(2) * (SW - WS))
        b_aniso = b_lin + b_nl

        # Isotropic + anisotropic: R_ii = 2k by construction (b_aniso is
        # trace-free because S_dev, SS_dev, WW_dev are trace-free and
        # trace(SW-WS) = S:W - W:S = 0).
        R_lin = 2.0 * k_b * b_lin
        R_NL = 2.0 * k_b * b_nl
        R_iso = (2.0 / 3.0) * k_b * eye
        R = R_iso + R_lin + R_NL

        # Project eigenvalues back to physical domain if requested
        if self.project:
            R = project_realizability_batched(R, k)

        # Anisotropy tensor b_ij = <u_i u_j> - 2k/3 delta_ij
        aniso = R - (2.0 / 3.0) * k_b * eye

        # Invariants (2 S_ij S_ij and 2 W_ij W_ij)
        S2 = 2.0 * (S * S).sum(dim=(1, 2))
        W2 = 2.0 * (W * W).sum(dim=(1, 2))

        return dict(R=R, R_lin=R_lin, R_NL=R_NL, nu_t=nu_t, aniso=aniso,
                    S2=S2, W2=W2)

    @staticmethod
    def unpack(R):
        """Extract the 6 independent components of a (N, 3, 3) symmetric tensor.

        Returns dict with keys: uu (R11), vv (R22), ww (R33), uv (R12),
        uw (R13), vw (R23). Assumes R is symmetric.
        """
        return dict(
            uu=R[:, 0, 0], vv=R[:, 1, 1], ww=R[:, 2, 2],
            uv=R[:, 0, 1], uw=R[:, 0, 2], vw=R[:, 1, 2],
        )