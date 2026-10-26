"""Step 2 — data self-consistency check (one-dimensional mean equilibrium).

Mean spanwise momentum of the z-averaged flow (mode 1):
    nu * d^2 W+/dy+^2 = d<vw>+/dy+ ,   W+(+-180) = 0  (no-slip walls)

We integrate the RHS twice on the DNS (non-uniform) y+ grid and fix the two
integration constants from the wall BCs, then compare the reconstructed
spanwise velocity with the W+ column of the data file. This quantifies how
self-consistent the DNS planar averages are, and validates the sign/column
conventions used by the PINN.

Run:  python -m pinn_tg.verify_1d
"""
from __future__ import annotations

import json
import os

import numpy as np

from .data import load_dns, RE_TAU


def cumtrapz(x, f):
    out = np.zeros_like(f)
    out[1:] = np.cumsum(0.5 * (f[:-1] + f[1:]) * np.diff(x))
    return out


def reconstruct_w(yp, vw):
    src = np.gradient(vw, yp)           # d<vw>+/dy+
    g1 = cumtrapz(yp, src)               # W' + c1
    g2 = cumtrapz(yp, g1)                # W  + c1*yp + c2
    # fix c1, c2 EXACTLY from W(yp[0]) = W(yp[-1]) = 0 (no-slip walls)
    M = np.array([[yp[0], 1.0], [yp[-1], 1.0]])
    coef = np.linalg.solve(M, -np.array([g2[0], g2[-1]]))
    return g2 + M[0, 0] * 0 + yp * coef[0] + coef[1], src


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_file = os.path.join(root, "data", "Retau180Ro30.txt")
    outdir = os.path.join(root, "outputs", "verify_1d")
    os.makedirs(outdir, exist_ok=True)

    dns, npts = load_dns(data_file)
    y = dns["y"]
    yp = y * RE_TAU
    W = dns["W"]
    vw = dns["vw"]

    Wmod, src = reconstruct_w(yp, vw)
    err = Wmod - W
    stats = {
        "n_points": int(npts),
        "max_abs_W": float(np.max(np.abs(W))),
        "rms_error": float(np.sqrt(np.mean(err**2))),
        "max_error": float(np.max(np.abs(err))),
        "rel_rms_error": float(np.sqrt(np.mean(err**2)) / np.max(np.abs(W))),
        "rel_max_error": float(np.max(np.abs(err)) / np.max(np.abs(W))),
        "max_abs_vw": float(np.max(np.abs(vw))),
        "triple_zero_W": [
            float(yp[i]) for i in range(1, len(W) - 1)
            if W[i - 1] * W[i + 1] < 0
        ],
    }
    # pointwise balance diagnostics: nu*W'' vs d<vw>/dy on raw grid
    Wpp = np.gradient(np.gradient(W, yp), yp)
    ok = (np.abs(yp) > 2.0)
    A, B = Wpp[ok], src[ok]
    slope = float(np.dot(A, B) / np.dot(B, B))
    corr = float(np.dot(A, B) / np.sqrt(np.dot(A, A) * np.dot(B, B)))
    resid = Wpp - src
    stats.update({
        "pointwise_slope": slope,
        "pointwise_corr": corr,
        "pointwise_rms_res_over_peak_src": float(
            np.sqrt(np.mean(resid**2)) / np.max(np.abs(src))),
        "bulk_res_over_peak": float(
            np.sqrt(np.mean(resid[np.abs(yp) > 20]**2)) / np.max(np.abs(src))),
        "nearwall_res_over_peak": float(
            np.sqrt(np.mean(resid[np.abs(yp) <= 20]**2)) / np.max(np.abs(src))),
    })
    with open(os.path.join(outdir, "verify_1d.json"), "w") as f:
        json.dump(stats, f, indent=2)

    # figure: reconstruction + balance source term
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.plot(W, yp, "o", ms=3, label="DNS  W$^+$ data")
    ax.plot(Wmod, yp, "-", lw=1.5,
            label=r"reconstructed: $\nu\,W^{+''}= \mathrm{d}\langle vw\rangle^+/\mathrm{d}y^+$")
    ax.set_xlabel(r"$W^+$")
    ax.set_ylabel(r"$y^+$")
    ax.legend(fontsize=8)
    ax.set_title(f"mode-1 equilibrium check (rel. RMS err = {stats['rel_rms_error']*100:.1f}%)")
    ax = axes[1]
    ax.plot(vw, yp, label=r"$\langle vw\rangle^+$")
    ax.plot(src, yp, label=r"$\mathrm{d}\langle vw\rangle^+/\mathrm{d}y^+$")
    ax.set_xlabel("value")
    ax.set_ylabel(r"$y^+$")
    ax.legend(fontsize=8)
    ax.set_title("source term profile")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "verify_1d.png"), dpi=160)

    print(json.dumps(stats, indent=2))
    print("saved:", outdir)


if __name__ == "__main__":
    main()
