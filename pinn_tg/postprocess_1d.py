"""Post-processing for the 1D PINN-RANS-ASM solver.

Loads a trained model (outputs/runs/<name>/model.pt), predicts fields on a
dense grid, compares with DNS (Ro_tau=30 case), checks tensor physicality,
and saves:
    fields.npz     y, U, W, k, eps, nu_t, R (6 components) for downstream 2D PINN
    summary.json   error metrics + physicality checks (with pass/fail flags)
    profiles.png   U, W, k, eps, nu_t + Reynolds stress comparison vs DNS

Usage:
    python -m pinn_tg.postprocess_1d --run rasm_1d_ro30
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch

from .pinn_1d import MLP1D, PINN1DConfig, RANSProblem1D

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, required=True, help="run name under outputs/runs/")
    ap.add_argument("--n-grid", type=int, default=401, help="dense prediction grid points")
    ap.add_argument("--no-dns", action="store_true", help="skip DNS comparison (non-Ro30 cases)")
    return ap.parse_args()


def load_run(run_name: str):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_dir = os.path.join(root, "outputs", "runs", run_name)
    ckpt = torch.load(os.path.join(run_dir, "model.pt"), map_location="cpu",
                      weights_only=False)
    cfgd = ckpt["cfg"]
    net = MLP1D(cfgd.get("width", 64), cfgd.get("depth", 4))
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    cfg = PINN1DConfig(
        Re_tau=ckpt.get("Re_tau", 180.0),
        Ro_tau=ckpt.get("Ro_tau", 30.0),
        width=cfgd.get("width", 64), depth=cfgd.get("depth", 4),
        w_pde=cfgd.get("w_pde", 1.0), w_bc=cfgd.get("w_bc", 50.0),
        w_real=cfgd.get("w_real", 10.0),
    )
    prob = RANSProblem1D(net, cfg)
    return prob, run_dir, ckpt


def predict(prob: RANSProblem1D, n_grid: int):
    # 绘图/导出坐标统一 float64（项目约定 #6）
    yhat = torch.linspace(-1.0, 1.0, n_grid, dtype=torch.float64).unsqueeze(-1)
    yhat32 = yhat.to(torch.float32)
    with torch.no_grad():
        out = prob.net(yhat32)
        U_raw, W_raw = out[:, 0:1], out[:, 1:2]
        logk, logeps = out[:, 2:3], out[:, 3:4]
        # 复现 fields() 的 wall mask（无梯度版本；与 pinn_1d 直接乘法一致）
        mask = (1.0 - yhat32 ** 2)
        U = (U_raw * mask)
        W = (W_raw * mask)
        k = (torch.exp(logk) * mask).clamp_min(1e-30)
        eps = torch.exp(logeps).clamp_min(1e-30)
        nu_t = prob.asm.C_mu * k.pow(2) / eps
        # ASM 张量（需要 S, W：无梯度近似——用有限差分求 dU/dy, dW/dy）
    # 有限差分导数（float64）
    U64, W64 = U.double().squeeze(-1).numpy(), W.double().squeeze(-1).numpy()
    y64 = yhat.squeeze(-1).numpy()
    dU = np.gradient(U64, y64)
    dW = np.gradient(W64, y64)
    n = len(y64)
    S = np.zeros((n, 3, 3), dtype=np.float64)
    Wt = np.zeros((n, 3, 3), dtype=np.float64)
    S[:, 0, 1] = S[:, 1, 0] = 0.5 * dU
    S[:, 1, 2] = S[:, 2, 1] = 0.5 * dW
    Om = prob.Omega_outer
    Wt[:, 0, 1] = 0.5 * dU
    Wt[:, 1, 0] = -0.5 * dU
    Wt[:, 1, 2] = 0.5 * dW - Om
    Wt[:, 2, 1] = -0.5 * dW + Om
    with torch.no_grad():
        asm_out = prob.asm(
            torch.tensor(S, dtype=torch.float32),
            torch.tensor(Wt, dtype=torch.float32),
            k[:, 0], eps[:, 0])
        R = asm_out["R"].numpy()  # (n, 3, 3)
    return dict(
        y=y64, U=U64, W=W64,
        k=k.double().squeeze(-1).numpy(),
        eps=eps.double().squeeze(-1).numpy(),
        nu_t=nu_t.double().squeeze(-1).numpy(),
        R=R,
    )


def physicality_checks(R: np.ndarray, k: np.ndarray):
    """Eigenvalue / symmetry / realness checks. Returns dict of metrics."""
    n = R.shape[0]
    sym_err = np.abs(R - np.transpose(R, (0, 2, 1))).max()
    real_ok = bool(np.all(np.isreal(R)))
    eig_min, eig_max = [], []
    viol = 0
    for i in range(n):
        e = np.linalg.eigvalsh(R[i])
        eig_min.append(e.min())
        eig_max.append(e.max())
        lo, hi = -2.0 * k[i] / 3.0, 4.0 * k[i] / 3.0
        if e.min() < lo - 1e-8 or e.max() > hi + 1e-8:
            viol += 1
    eig_min = np.array(eig_min)
    eig_max = np.array(eig_max)
    # 特征值与界限的平均相对越界
    lo_arr = -2.0 * k / 3.0
    hi_arr = 4.0 * k / 3.0
    viol_lo = np.maximum(0.0, (lo_arr - eig_min) / np.abs(lo_arr).clip(1e-30))
    viol_hi = np.maximum(0.0, (eig_max - hi_arr) / np.abs(hi_arr).clip(1e-30))
    return dict(
        symmetry_max_err=float(sym_err),
        real_ok=real_ok,
        eig_violation_frac=float(viol / n),
        eig_lo_viol_mean=float(viol_lo.mean()),
        eig_hi_viol_mean=float(viol_hi.mean()),
    )


def compare_dns(fields: dict, run_dir: str):
    """Compare PINN profiles with DNS (Ro_tau=30 only). Returns metrics dict."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dns_path = os.path.join(root, "data", "Retau180Ro30.txt")
    if not os.path.exists(dns_path):
        return None
    from .data import load_dns, RE_TAU
    dns, npts = load_dns(dns_path)
    y_dns = dns["y"]
    # 插值 DNS 到 PINN 网格
    y = fields["y"]

    def interp(v):
        return np.interp(y, y_dns, v)

    # DNS 量都在 wall units（u_tau 速度单位）；PINN 速度同为 u_tau 单位，直接可比
    # k/eps 应力是速度平方单位 u_tau^2：PINN 输出也是 u_tau 单位制
    metrics = {}
    pairs = [("U", "U"), ("W", "W"), ("k", "TKE")]
    for pin, dns_name in pairs:
        a = fields[pin]
        b = interp(dns[dns_name])
        # 排除壁面 mask 区（k 在壁面被强制为 0，DNS 壁面也是 0）
        scale = max(np.abs(b).max(), 1e-30)
        rms = float(np.sqrt(np.mean((a - b) ** 2)) / scale)
        metrics[f"rmse_rel_{pin}"] = rms
    # 应力分量（单位换算：R_PINN 在 u_tau^2·? —— 需检查量纲一致性）
    # PINN 的 R 来自 ASM(S, W, k, eps)，全部在同一单位制
    R = fields["R"]
    k_interp = fields["k"]
    scale_k = max(k_interp.max(), 1e-30)
    for name, idx in [("uu", (0, 0)), ("vv", (1, 1)), ("ww", (2, 2)),
                      ("uv", (0, 1)), ("uw", (0, 2)), ("vw", (1, 2))]:
        a = R[:, idx[0], idx[1]]
        b = interp(dns[name])
        # 应力分量尺度用 DNS 分量自身量级
        scale = max(np.abs(b).max(), 1e-30)
        rms = float(np.sqrt(np.mean((a - b) ** 2)) / scale)
        metrics[f"rmse_rel_R_{name}"] = rms
    return metrics


def make_figure(fields: dict, run_dir: str, dns_metrics=None):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dns_path = os.path.join(root, "data", "Retau180Ro30.txt")
    has_dns = os.path.exists(dns_path) and dns_metrics is not None
    if has_dns:
        from .data import load_dns
        dns, _ = load_dns(dns_path)

    y = fields["y"]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharey=True)
    # Row 1: U, W, k, eps
    items1 = [("U", fields["U"], "U / u_tau"),
              ("W", fields["W"], "W / u_tau"),
              ("k", fields["k"], "k / u_tau^2"),
              ("eps", fields["eps"], "eps (PINN units)")]
    dns_keys = {"U": "U", "W": "W", "k": "TKE", "eps": None}
    for ax, (name, v, lab) in zip(axes[0], items1):
        ax.plot(v, y, "b-", lw=2, label="PINN-ASM")
        if has_dns and dns_keys[name]:
            ax.plot(dns[dns_keys[name]], dns["y"], "k--", lw=1,
                    label="DNS")
        ax.set_title(lab)
        ax.set_xlabel(name)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0, 0].set_ylabel("y / h")
    # Row 2: nu_t + 3 stress components
    axes[1, 0].plot(fields["nu_t"], y, "b-", lw=2, label="PINN")
    axes[1, 0].set_title("nu_t")
    axes[1, 0].set_xlabel("nu_t")
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].set_ylabel("y / h")
    R = fields["R"]
    stress = [("uu", R[:, 0, 0], "uu"), ("vw", R[:, 1, 2], "vw"),
              ("uv", R[:, 0, 1], "uv")]
    for ax, (name, v, lab) in zip(axes[1, 1:], stress):
        ax.plot(v, y, "b-", lw=2, label="PINN-ASM")
        if has_dns:
            ax.plot(dns[name], dns["y"], "k--", lw=1, label="DNS")
        ax.set_title(f"<{lab}>")
        ax.set_xlabel(f"<{lab}>")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("1D PINN-RANS-ASM profiles vs DNS (Ro_tau=30)")
    fig.tight_layout()
    out = os.path.join(run_dir, "profiles.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main():
    args = parse_args()
    prob, run_dir, ckpt = load_run(args.run)
    fields = predict(prob, args.n_grid)

    # physicality
    phys = physicality_checks(fields["R"], fields["k"])
    # DNS comparison
    dns_metrics = None
    if not args.no_dns and abs(ckpt.get("Ro_tau", 30.0) - 30.0) < 1e-6:
        dns_metrics = compare_dns(fields, run_dir)

    # save npz (downstream 2D PINN input)
    npz_path = os.path.join(run_dir, "fields.npz")
    R = fields["R"]
    np.savez(
        npz_path,
        y=fields["y"], U=fields["U"], W=fields["W"],
        k=fields["k"], eps=fields["eps"], nu_t=fields["nu_t"],
        uu=R[:, 0, 0], vv=R[:, 1, 1], ww=R[:, 2, 2],
        uv=R[:, 0, 1], uw=R[:, 0, 2], vw=R[:, 1, 2],
        Ro_tau=ckpt.get("Ro_tau", 30.0), Re_tau=ckpt.get("Re_tau", 180.0),
    )

    fig_path = make_figure(fields, run_dir, dns_metrics)

    summary = dict(
        run=args.run,
        Ro_tau=ckpt.get("Ro_tau"), Re_tau=ckpt.get("Re_tau"),
        physicality=phys,
        dns_comparison=dns_metrics,
    )
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # console report
    print(f"=== postprocess: {args.run} ===")
    print(f"physicality:")
    print(f"  symmetry max err : {phys['symmetry_max_err']:.2e}  "
          f"{'PASS' if phys['symmetry_max_err'] < 1e-6 else 'FAIL'}")
    print(f"  real            : {phys['real_ok']}")
    print(f"  eig violation   : {phys['eig_violation_frac']*100:.1f}%")
    if dns_metrics:
        print(f"DNS comparison (relative RMS):")
        for kk, vv in dns_metrics.items():
            flag = "OK " if vv < 0.05 else ("WARN" if vv < 0.20 else "BAD ")
            print(f"  {flag} {kk:20s}: {vv:.4f}")
    print(f"saved: {npz_path}")
    print(f"saved: {fig_path}")
    print(f"saved: {os.path.join(run_dir, 'summary.json')}")


if __name__ == "__main__":
    main()