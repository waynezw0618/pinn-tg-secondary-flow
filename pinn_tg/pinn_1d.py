"""pinn_tg/pinn_1d.py - 1D PINN-RANS-ASM solver for rotating channel flow.

Solves the one-dimensional steady RANS equations for the streamwise-rotating
turbulent channel flow (Yang & Wang 2018 Case C), with the Reynolds stress
tensor closed by the Wallin-Johansson (2000) explicit algebraic stress model
(Gatski-Speziale 1993 / Craft-Launder-Suga 1996 quadratic NLEVM equivalent).

Coordinate convention: x = streamwise (axis 0), y = wall-normal (axis 1),
z = spanwise (axis 2). Channel half-height h = 1 (normalized). Outer units:
length by h, velocity by u_tau, so nu_outer = 1/Re_tau.

Network outputs (4 scalars at each y in [-1, +1]):
    U(y)        mean streamwise velocity
    Wbar(y)     mean spanwise velocity (mode 1 secondary flow)
    log k(y)    log turbulent kinetic energy (TKE = k)
    log eps(y)  log turbulent dissipation rate

From these, we compute:
    S_ij, W_ij    strain-rate / rotation-rate tensors (1D channel: only y-gradient)
    nu_t = C_mu * k * tau     eddy viscosity (tau = k / eps)
    R_ij = ASM(S, W, k, eps)   closed Reynolds stress tensor
    P_k = nu_t * |S|^2         turbulent production

RANS residuals (steady, 1D):
    R_U   = d/dy [(nu + nu_t) dU/dy] - d<uv>/dy           (momentum x; with -dP/dx folded into bc)
    R_W   = d/dy [(nu + nu_t) dWbar/dy] - d<vw>/dy + 2*Omega*U  (momentum z; Coriolis)
    R_k   = P_k - eps + d/dy [(nu + nu_t/sigma_k) dk/dy] - 2*Omega*<uw>   (TKE transport; Launder rotation)
    R_eps = (C_eps1*P_k - C_eps2*eps)*T^{-1} + d/dy[(nu + nu_t/sigma_eps) d eps/dy]
            + ... Launder rotation correction term         (eps transport)

Boundary conditions (high-Re / wall-function k-eps consistency):
    y = +/- 1:  U = Wbar = 0          (no-slip, enforced exactly by the mask)
    y = +/- 1:  d k / d y   = 0       (zero-gradient TKE)
    y = +/- 1:  d eps / d y = 0       (dissipation equilibrium)

Loss components (all dimensionless / scale-invariant where needed):
    L_pde   : scale-invariant (relative) L2 of (R_U, R_W, R_k, R_eps)
    L_bc    : boundary residual (U, Wbar, dk/dy, deps/dy at y=+/-1)
    L_real  : realizability (R_ij eigenvalues in [-2k/3, 4k/3])
    L_smooth: optional smoothness penalty on d log k / dy etc.

Coordinate / units:
    y in [-1, +1] -> y+ in [-180, +180] -> y_outer same; nu_outer = 1/180.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .asm import ASM


# k-epsilon model constants (standard Launder-Jones 1974 + Launder rotation)
C_MU = 0.09
C_EPS1 = 1.44
C_EPS2 = 1.92
SIGMA_K = 1.0
SIGMA_EPS = 1.3
# Launder rotation correction (Energy equation paper): drag on k due to rotation
# adds -2*Omega*<uw>; for eps we use the standard form (no extra correction
# here, since rotating-channel calibration typically uses an additional
# C_EPS1(Ro) modulation -- kept out for simplicity).

NU_OUTER = 1.0 / 180.0  # outer-unit kinematic viscosity at Re_tau=180


# ------------------ Network ------------------


class MLP1D(nn.Module):
    """Simple 1D MLP: scalar input -> 4 scalar outputs (U, W, log k, log eps).

    Width and depth configurable. Tanh activations, Xavier initialization.
    """

    def __init__(self, width: int = 64, depth: int = 4):
        super().__init__()
        layers = [nn.Linear(1, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 4)]  # [U, W, log_k, log_eps]
        self.net = nn.Sequential(*layers)
        # Xavier init for hidden layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        # 物理先验缩放：让输出层初值落在 DNS 典型范围（外尺度 u_tau 单位）
        # U, W ~ O(0.1-1); k ~ 2 (k+ 峰值 2-3.5); eps ~ 50 (eps_PINN=180*eps+)
        # 由此 nu_t = C_mu k^2/eps ~ 0.09*4/50 = 0.0072 (混合区合理值)
        last = self.net[-1]
        nn.init.normal_(last.weight, mean=0.0, std=0.1)
        with torch.no_grad():
            last.bias[0] = 0.0         # U
            last.bias[1] = 0.0         # W
            last.bias[2] = 0.6931      # log k, k~2
            last.bias[3] = 3.9120      # log eps, eps~50

    def forward(self, yhat: torch.Tensor) -> torch.Tensor:
        """Return (N, 4) tensor with [U, W, log_k, log_eps]."""
        return self.net(yhat)


# ------------------ PDE residuals ------------------


def _dyd_dyhat():
    """dy/dyhat = 1 (since y = h * yhat and we normalized h=1)."""
    return 1.0


def _first_deriv(out: torch.Tensor, yhat: torch.Tensor,
                 col: int, create: bool = True) -> torch.Tensor:
    """Compute d(out[col]) / dyhat via autograd (yhat is the only network input).

    Returns shape (N, 1).
    """
    g = torch.autograd.grad(
        out[:, col:col + 1].sum(), yhat,
        create_graph=create, retain_graph=True,
    )[0]
    return g  # shape (N, 1)


def _second_deriv(out: torch.Tensor, yhat: torch.Tensor,
                  col: int, create: bool = True) -> torch.Tensor:
    """Compute d^2(out[col]) / dyhat^2."""
    g1 = _first_deriv(out, yhat, col, create=create)
    g2 = torch.autograd.grad(
        g1.sum(), yhat,
        create_graph=create, retain_graph=True,
    )[0]
    return g2


@dataclass
class PINN1DConfig:
    """Hyperparameters for the 1D PINN-RANS-ASM solver."""
    Re_tau: float = 180.0
    Ro_tau: float = 30.0         # Ro_tau = 2*Omega*h/u_tau -> Omega_outer = Ro_tau/2
    width: int = 64
    depth: int = 4
    # Loss weights (relative; L_pde is the scale-invariant baseline)
    w_pde: float = 1.0
    w_bc: float = 50.0
    w_real: float = 10.0
    w_smooth: float = 0.01
    w_data: float = 100.0  # U, W 数据锚定（破 PDE 齐次平凡解）


class RANSProblem1D:
    """1D RANS-ASM PINN problem assembly. Holds the network and ASM closure,
    provides methods to compute PDE residuals, boundary loss, realizability loss.
    """

    def __init__(self, net: MLP1D, cfg: PINN1DConfig,
                 asm_instance: Optional[ASM] = None):
        self.net = net
        self.cfg = cfg
        self.asm = asm_instance or ASM(C_mu=C_MU, project=True)
        self.Omega_outer = cfg.Ro_tau / 2.0
        self.nu_outer = 1.0 / cfg.Re_tau

    # ---------- Field extraction (post-enforce physical constraints) ----------

    @staticmethod
    def _apply_wall_mask(U_raw, W_raw, logk_raw, logeps_raw, yhat):
        """Enforce U = Wbar = 0 at the walls via a (1 - yhat^2) hard mask.

        NOTE on k / eps: for the high-Re (wall-function) k-epsilon model the
        wall condition is *zero gradient*, NOT k = 0. Forcing k -> 0 at the
        wall makes tau = k/eps -> 0 and the eps source
        (C_e1 P_k - C_e2 eps)/tau -> singular. Measured at init
        (2026-09-29): with the k-mask in place L_eps ~ 1.07e4 while L_U,L_W ~ O(1).
        Removing the k-mask and enforcing dk/dy = deps/dy = 0 at the wall
        (loss_bc) makes the residual O(1). k, eps stay strictly positive via
        exp(log).

        The U-mask fixes U(+/-1) = 0 only; the wall slope stays free because
        dU/dy|_wall = -2*U_raw(+/-1).  (Earlier the tensor builder differentiated
        the *unmasked* net output, which is inconsistent with the masked field
        used everywhere else -- fixed in _tensors.)
        """
        wall_mask = (1.0 - yhat ** 2)  # zero at yhat = +/-1
        U = U_raw * wall_mask
        W = W_raw * wall_mask
        k = torch.exp(logk_raw).clamp_min(1e-12)
        eps = torch.exp(logeps_raw).clamp_min(1e-12)
        return U, W, k, eps

    def fields(self, yhat: torch.Tensor):
        """Return [U, W, k, eps, nu_t, R_full] given yhat (N, 1) in [-1, +1].

        U, W, k satisfy no-slip + zero TKE at walls via multiplicative mask.
        """
        out = self.net(yhat)
        U_raw = out[:, 0:1]
        W_raw = out[:, 1:2]
        logk_raw = out[:, 2:3]
        logeps_raw = out[:, 3:4]
        U, W, k, eps = self._apply_wall_mask(
            U_raw, W_raw, logk_raw, logeps_raw, yhat)
        nu_t = self.asm.C_mu * k.pow(2) / eps  # (N,1)
        return {
            "U": U, "W": W, "k": k, "eps": eps, "nu_t": nu_t,
            "raw_U": U_raw, "raw_W": W_raw,
            "raw_logk": logk_raw, "raw_logeps": logeps_raw,
        }

    # ---------- Strain & rotation tensors (1D channel, only y-gradient) ----------

    def _tensors(self, yhat: torch.Tensor, fields: dict):
        """Compute S_ij, W_ij (N, 3, 3) for a 1D channel.

        Only the y-derivative is non-zero.  With x=axis0, y=axis1, z=axis2 and
        (U, V, W) = (U(y), 0, Wbar(y)):

            S_ij = (1/2)(du_i/dx_j + du_j/dx_i)
              S_01 = S_10 = (1/2) dU/dy
              S_12 = S_21 = (1/2) dWbar/dy

            R_ij = (1/2)(du_i/dx_j - du_j/dx_i)      (mean-flow rotation)
              R_01 = -R_10 = (1/2) dU/dy
              R_12 = (1/2)(dV/dz - dWbar/dy) = -(1/2) dWbar/dy      <-- sign
              R_21 = +(1/2) dWbar/dy

            system rotation along x:  R^rot_12 = -Omega,  R^rot_21 = +Omega

        SIGN FIX (2026-09-29): the mean-flow part of R_12 previously carried the
        wrong sign (+(1/2)dWbar/dy); with |dWbar/dy| ~ 1 and Omega ~ 15 the
        error is only ~5%, but it matters for <vw> (which is generated by the
        W-square term).  The correct combination is
              W_12 = -(1/2) dWbar/dy - Omega,   W_21 = +(1/2) dWbar/dy + Omega.
        """
        # IMPORTANT: differentiate the *masked* fields (fields["U"], fields["W"]),
        # not the raw net output.  Only the masked field is the physical velocity
        # (U = U_raw (1-yhat^2)); using d(net)/dy has the wrong wall slope and is
        # inconsistent with the momentum-flux term below.
        dU_dy = torch.autograd.grad(
            fields["U"].sum(), yhat, create_graph=True, retain_graph=True)[0]
        dW_dy = torch.autograd.grad(
            fields["W"].sum(), yhat, create_graph=True, retain_graph=True)[0]

        N = yhat.shape[0]
        S = torch.zeros(N, 3, 3, dtype=yhat.dtype, device=yhat.device)
        W_ = torch.zeros(N, 3, 3, dtype=yhat.dtype, device=yhat.device)
        S[:, 0, 1] = S[:, 1, 0] = 0.5 * dU_dy[:, 0]
        S[:, 1, 2] = S[:, 2, 1] = 0.5 * dW_dy[:, 0]
        W_[:, 0, 1] = 0.5 * dU_dy[:, 0]
        W_[:, 1, 0] = -0.5 * dU_dy[:, 0]
        W_[:, 1, 2] = -0.5 * dW_dy[:, 0] - self.Omega_outer
        W_[:, 2, 1] = 0.5 * dW_dy[:, 0] + self.Omega_outer
        return S, W_, dU_dy, dW_dy

    # ---------- PDE residuals ----------

    def pde_residuals(self, yhat: torch.Tensor):
        """Compute 4 RANS PDE residuals + diagnostics. All in OUTER units.

        R_U     = d/dy [(nu + nu_t) dU/dy] - d<uv>/dy
                  (note: -1/rho dP/dx is implicit in the chosen U profile;
                   we keep R_U as a residual that the network can drive to 0)
        R_W     = d/dy [(nu + nu_t) dWbar/dy] - d<vw>/dy + 2*Omega*U
        R_k     = P_k - eps + d/dy [(nu + nu_t/sigma_k) dk/dy] - 2*Omega*<uw>
        R_eps   = (C_eps1*P_k - C_eps2*eps)*T^{-1} + d/dy[(nu + nu_t/sigma_eps) d eps/dy]

        Returns:
            dict with keys 'R_U', 'R_W', 'R_k', 'R_eps' (each (N, 1))
                 plus diagnostics 'P_k', 'nu_t', 'R_full', 'tau', 'T'
        """
        fields = self.fields(yhat)
        U, W_, k, eps, nu_t = fields["U"], fields["W"], fields["k"], fields["eps"], fields["nu_t"]

        # Strain and rotation
        S, Wt, dU_dy, dW_dy = self._tensors(yhat, fields)
        asm_out = self.asm(S, Wt, k[:, 0], eps[:, 0])
        R = asm_out["R"]  # (N, 3, 3)
        uv = R[:, 0, 1]
        vw = R[:, 1, 2]
        uw = R[:, 0, 2]

        # Production P_k = -R_ij S_ij（ASM 应力直接收缩，含非线性贡献；
        # 各向同性部分 (2k/3)delta_ij 与无迹 S 收缩为零，自洽）
        P_k = -(R * S).sum(dim=(1, 2))  # (N,)

        # tau = k/eps, T = k/eps (same thing in this convention)
        # clamp 防 NaN + 物理范围：DNS 平衡下 tau = k/eps ~ 0.04 (k~2, eps~50)
        # 即使强不平衡 tau 也不会超过 O(1)；clamp(1e-3, 10) 覆盖 4 个量级余量
        eps_safe = eps[:, 0].clamp_min(1e-30)
        k_safe = k[:, 0].clamp_min(1e-30)
        Tinv = (eps_safe / k_safe).clamp(1e-2, 1e2)
        tau = (k_safe / eps_safe).clamp(1e-2, 1e2)

        # ---- R_U (streamwise momentum) ----
        # ASM 封闭下的动量方程（扩散项只用分子粘性 ν；涡粘贡献已在
        # ASM 应力的线性部分 -nu_t*S 中，用 (nu+nu_t) 会重复计数）:
        #   0 = -dP/dx + d/dy[nu dU/dy] - d<uv>_ASM/dy
        # 压力梯度 dP/dx = 1（摩擦驱动槽道流，tau_w=1 归一化）。
        # 没有此项方程齐次，U=k=eps=0 是平凡解（实测网络坍缩到零场）。
        flux_U = self.nu_outer * dU_dy[:, 0]  # (N,)
        d_flux_U = torch.autograd.grad(
            flux_U.sum(), yhat, create_graph=True, retain_graph=True)[0]
        duv_dy = torch.autograd.grad(uv.sum(), yhat, create_graph=True,
                                     retain_graph=True)[0][:, 0]
        R_U = d_flux_U[:, 0] - duv_dy + 1.0  # (N,)

        # ---- R_W (spanwise momentum, mode 1) ----
        # verify_1d.py 验证的 DNS 自洽形式: nu d^2 W/dy^2 = d<vw>/dy。
        # Omega 沿 x 与 U 平行 -> 平均科氏力 2*Omega x Ubar = 0，
        # 无 2*Omega*U 项（该形式属于 spanwise-rotation 槽道，此处不适用）。
        # W_bar 由 ASM <vw> 的梯度间接驱动（旋转经 ASM 的 Omega* 进入应力）。
        flux_W = self.nu_outer * dW_dy[:, 0]
        d_flux_W = torch.autograd.grad(
            flux_W.sum(), yhat, create_graph=True, retain_graph=True)[0]
        dvw_dy = torch.autograd.grad(vw.sum(), yhat, create_graph=True,
                                     retain_graph=True)[0][:, 0]
        R_W = d_flux_W[:, 0] - dvw_dy

        # ---- R_k (TKE transport) ----
        # Yang & Wang 2018 Table 2: Coriolis terms C11+C22+C33 = 0,
        # 科氏力对 TKE 的净贡献恒为零（旋转不做功），不加 -2*Omega*<uw>。
        # 旋转效应经 ASM 的 Omega*（绝对涡量进入非线性系数）影响应力，
        # 进而影响 P_k 和 k 输运。
        # Diffusive: d/dy [(nu + nu_t/sigma_k) dk/dy]
        dk_dy = torch.autograd.grad(k[:, 0].sum(), yhat, create_graph=True,
                                    retain_graph=True)[0][:, 0]
        nu_eff_k = self.nu_outer + nu_t[:, 0] / SIGMA_K
        flux_k = nu_eff_k * dk_dy
        d_flux_k = torch.autograd.grad(
            flux_k.sum(), yhat, create_graph=True, retain_graph=True)[0]
        # d/dy[(nu + nu_t/sigma_k) dk/dy] has shape (N,)
        R_k = P_k - eps[:, 0] + d_flux_k[:, 0]

        # ---- R_eps (dissipation transport) ----
        # Diffusive: d/dy [(nu + nu_t/sigma_eps) d eps/dy]
        deps_dy = torch.autograd.grad(eps[:, 0].sum(), yhat, create_graph=True,
                                      retain_graph=True)[0][:, 0]
        nu_eff_eps = self.nu_outer + nu_t[:, 0] / SIGMA_EPS
        flux_eps = nu_eff_eps * deps_dy
        d_flux_eps = torch.autograd.grad(
            flux_eps.sum(), yhat, create_graph=True, retain_graph=True)[0]
        source_eps = (C_EPS1 * P_k - C_EPS2 * eps[:, 0]) * Tinv
        R_eps = source_eps + d_flux_eps[:, 0]

        return {
            "R_U": R_U.unsqueeze(-1), "R_W": R_W.unsqueeze(-1),
            "R_k": R_k.unsqueeze(-1), "R_eps": R_eps.unsqueeze(-1),
            "P_k": P_k, "nu_t": nu_t, "R_full": R, "tau": tau,
            "T": k[:, 0] / eps[:, 0],
            "uv": uv, "vw": vw, "uw": uw,
            # --- term magnitudes for relative (scale-invariant) normalization ---
            "d_flux_U": d_flux_U[:, 0], "duv_dy": duv_dy,
            "d_flux_W": d_flux_W[:, 0], "dvw_dy": dvw_dy,
            "d_flux_k": d_flux_k[:, 0],
            "d_flux_eps": d_flux_eps[:, 0], "source_eps": source_eps,
            "eps_": eps[:, 0],
        }

    # ---------- Loss components ----------

    def loss_pde(self, yhat: torch.Tensor, eps: float = 1e-30) -> torch.Tensor:
        """Scale-invariant (relative) PDE residual loss.

        上一版用固定的 DNS 典型尺度 (1.0 / 0.01 / 2.0 / 50.0) 归一化：网络
        初始化时离 DNS 很远，残差被放大到 1e4 量级（实测 L_eps ~ 1.07e4），
        训练被单一方程主导。现改为**逐点相对残差**：

            r = R / (|该方程各项量级之和| + eps_small)

        使每一项的贡献恒为 O(1)，与网络当前所在解分支无关（符合项目约定 #2
        "PDE 损失必须尺度不变归一化"）。物理含义：衡量方程被违反的*相对*程度。

        Scales:
            U  : 压力驱动 = 1  + |d<uv>/dy|
            W  : |d(nu dW/dy)/dy| + |d<vw>/dy|
            k  : |P_k| + eps + |d(dk flux)/dy|
            eps: |source_eps| + |d(d eps flux)/dy|
        """
        res = self.pde_residuals(yhat)

        den_U = 1.0 + res["duv_dy"].abs()
        den_W = 1e-8 + res["d_flux_W"].abs() + res["dvw_dy"].abs()
        den_k = 1e-8 + res["P_k"].abs() + res["eps_"] + res["d_flux_k"].abs()
        den_eps = 1e-8 + res["source_eps"].abs() + res["d_flux_eps"].abs()

        r_U_norm = (res["R_U"][:, 0] / den_U).clamp(-1e4, 1e4)
        r_W_norm = (res["R_W"][:, 0] / den_W).clamp(-1e4, 1e4)
        r_k_norm = (res["R_k"][:, 0] / den_k).clamp(-1e4, 1e4)
        r_eps_norm = (res["R_eps"][:, 0] / den_eps).clamp(-1e4, 1e4)

        L_U = (r_U_norm ** 2).mean()
        L_W = (r_W_norm ** 2).mean()
        L_k = (r_k_norm ** 2).mean()
        L_eps = (r_eps_norm ** 2).mean()
        return L_U + L_W + L_k + L_eps

    def loss_bc(self, yhat_walls: torch.Tensor) -> torch.Tensor:
        """Boundary loss at y = +/-1.

        High-Re (wall-function) k-epsilon wall conditions:
            U = Wbar = 0          (no-slip; enforced exactly by the mask, so the
                                   penalty below only checks the mask is active)
            d k / d y   = 0       (zero-gradient TKE)
            d eps / d y = 0       (dissipation equilibrium)
        """
        f = self.fields(yhat_walls)
        dk_dy = torch.autograd.grad(
            f["k"].sum(), yhat_walls, create_graph=True, retain_graph=True)[0]
        deps_dy = torch.autograd.grad(
            f["eps"].sum(), yhat_walls, create_graph=True, retain_graph=True)[0]
        # U, W vanish by mask construction; penalize their magnitude as a
        # numerical self-check (should already be ~0).
        return ((dk_dy ** 2).mean() + (deps_dy ** 2).mean()
                + (f["U"] ** 2).mean() + (f["W"] ** 2).mean())

    def loss_realizability(self, yhat: torch.Tensor) -> torch.Tensor:
        """Soft penalty on Reynolds-stress eigenvalue violations.

        For each y, project R onto the realizability domain [-2k/3, 4k/3].
        Penalty = mean of squared violation magnitudes (clipped to >=0).
        """
        fields = self.fields(yhat)
        k = fields["k"][:, 0]
        res = self.pde_residuals(yhat)
        R = res["R_full"]
        # Eigenvalue test
        eig_lo = (-2.0 / 3.0) * k
        eig_hi = (4.0 / 3.0) * k
        # batched eigvalsh（可微），NaN 安全
        idx = torch.randperm(R.shape[0])[:64]  # subsample for memory
        R_sub = R[idx]
        k_sub = k[idx]
        try:
            eigs = torch.linalg.eigvalsh(R_sub)  # (n, 3)
            # nan_to_num 防退化特征值反传 NaN
            eigs = torch.nan_to_num(eigs, nan=0.0, posinf=0.0, neginf=0.0)
            pen_lo = torch.relu(eig_lo[idx].unsqueeze(-1) - eigs).pow(2).sum(dim=1)
            pen_hi = torch.relu(eigs - eig_hi[idx].unsqueeze(-1)).pow(2).sum(dim=1)
            penalty = (pen_lo + pen_hi).mean()
        except Exception:
            penalty = torch.tensor(0.0, dtype=R.dtype, device=R.device)
        return penalty

    def loss_data(self, yhat_data: torch.Tensor,
                 U_dns: torch.Tensor, W_dns: torch.Tensor) -> torch.Tensor:
        """Data loss on U and W_bar (DNS anchor).

        Why mandatory: 1D RANS-ASM is underdetermined without anchor.
        PDE 残差对 (k, eps) 齐次，trivial 解 k=eps=0, U=90(1-yhat^2) 满足全部
        4 个 RANS 方程（已实测：PINN 不加数据损失会收敛到此分支，U_max=115
        vs DNS U+~17）。U, W 锁数据 + k, eps 自由 → ASM 应力由 ASM 公式从
        (S, W, k, eps) 自洽得出 —— 仍属"完全脱开 DNS 应力"（Layer 2 接收）。
        """
        out = self.net(yhat_data)
        # wall-mask 已强制 U,W=0 在壁面：仅对 |yhat|<1 内部点有意义
        mask = (1.0 - yhat_data ** 2)
        U = out[:, 0:1] * mask
        W = out[:, 1:2] * mask
        # 尺度：用 DNS U, W 自身量级，避免量级差主导
        U_scale = U_dns.abs().max().clamp_min(1.0)
        W_scale = W_dns.abs().max().clamp_min(1.0)
        l_U = ((U[:, 0] - U_dns[:, 0]) / U_scale).pow(2).mean()
        l_W = ((W[:, 0] - W_dns[:, 0]) / W_scale).pow(2).mean()
        return l_U + l_W

    def loss(self, yhat_collocation: torch.Tensor,
             yhat_walls: torch.Tensor,
             yhat_data: torch.Tensor = None,
             U_dns: torch.Tensor = None,
             W_dns: torch.Tensor = None,
             w_pde: float = None, w_bc: float = None,
             w_real: float = None, w_data: float = None) -> dict:
        """Compute the total loss and break down each component."""
        if w_pde is None: w_pde = self.cfg.w_pde
        if w_bc is None: w_bc = self.cfg.w_bc
        if w_real is None: w_real = self.cfg.w_real
        if w_data is None: w_data = self.cfg.w_data
        L_pde = self.loss_pde(yhat_collocation)
        L_bc = self.loss_bc(yhat_walls)
        L_real = self.loss_realizability(yhat_collocation)
        L_data = torch.tensor(0.0)
        if yhat_data is not None and U_dns is not None and W_dns is not None:
            L_data = self.loss_data(yhat_data, U_dns, W_dns)
        L_total = (w_pde * L_pde + w_bc * L_bc
                   + w_real * L_real + w_data * L_data)
        return dict(
            total=L_total, pde=L_pde, bc=L_bc, real=L_real, data=L_data,
            w_pde=w_pde, w_bc=w_bc, w_real=w_real, w_data=w_data,
        )


__all__ = ["MLP1D", "PINN1DConfig", "RANSProblem1D",
           "C_MU", "C_EPS1", "C_EPS2", "SIGMA_K", "SIGMA_EPS", "NU_OUTER"]