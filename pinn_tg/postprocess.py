"""Post-processing: (y,z)-plane velocity vectors, vorticity contours,
z-mean profile checks and spanwise spectra for a trained TG-vortex PINN.

Run:  python -m pinn_tg.postprocess --run beta0.05_lam1.5_seed0
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from .data import load_dns, build_profiles, RE_TAU
from .checkpoint import load_run_net
from .pinn import TGProblem


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, required=True, help="run folder name")
    ap.add_argument("--n-lam", type=int, default=2, help="wavelengths to plot")
    ap.add_argument("--ny", type=int, default=289)
    ap.add_argument("--nz", type=int, default=256)
    return ap.parse_args()


def load_run(root, name):
    outdir = os.path.join(root, "outputs", "runs", name)
    net, cfg, _structure = load_run_net(outdir)
    return net, cfg, outdir


def main():
    args = parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    net, cfg, outdir = load_run(root, args.run)

    datafile = cfg.get("data", "Retau180Ro30.txt")
    dns, _ = load_dns(os.path.join(root, "data", datafile))
    # If the run was driven by an ASM <vv> profile (train.py --asm-npz), use
    # that same profile here so the reported L_energy target matches training.
    # W̄ and nu_t are the DNS ones in all our runs, so the vorticity fields are
    # unaffected either way.
    asm_npz = cfg.get("asm_npz", "")
    if asm_npz:
        from .data import load_asm_profiles
        ap_ = asm_npz if os.path.isabs(asm_npz) else os.path.join(root, asm_npz)
        profiles = load_asm_profiles(ap_)
        profiles["U"] = None
        print(f"postprocess: using ASM profile {ap_}")
    else:
        profiles = build_profiles(dns)
    prob = TGProblem(profiles, lam=cfg["lam"], ny=cfg["ny"], nz=cfg["nz"],
                     beta=cfg["beta"], use_nut=not cfg.get("no_nut", False),
                     use_aniso=not cfg.get("no_aniso", False))

    lam = cfg["lam"]
    ny, nz = args.ny, args.nz * args.n_lam
    grid, shape = prob.make_grid(ny, nz, zmax=lam * args.n_lam)
    f = prob.fields(net, grid=grid, shape=shape)
    _, _, diag, _ = prob.losses(net, grid=grid, shape=shape)

    Y = grid[0].view(ny, nz).detach().numpy()
    Z = grid[1].view(ny, nz).detach().numpy()
    # exact float64 spanwise coordinates for streamplot: the float32
    # training-style z grid has ulp-level non-uniform spacing whenever
    # zmax is not a dyadic value (e.g. lam=2.4 -> zmax=4.8), which trips
    # matplotlib's equally-spaced check.
    zrow = np.linspace(0.0, lam * args.n_lam, nz)

    def fld(key):
        return f[key].detach().view(ny, nz).numpy()

    V = fld("V")
    Wt = fld("Wt")
    Wtot = fld("W")
    Om = fld("Om")
    R = fld("R")
    Ps = fld("phi")
    y1 = Y[:, 0].copy()

    # ------------------------------------------------------------------ plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir = os.path.join(outdir, "figures")
    os.makedirs(figdir, exist_ok=True)

    # --- Fig 1: velocity vector field in the (y, z) plane (user request) ---
    s = 5  # vector subsampling
    qscale = float(np.hypot(V, Wt).max()) / 0.10   # longest arrow ~0.10 h
    # Component convention (verified numerically): the ansatz gives
    # v~ = dphi/dz (wall-normal) and w~ = -dphi/dy (spanwise), so in the
    # (z, y) plotting plane the vector is (U_along_z, V_along_y) = (Wt, V).
    # Passing (V, Wt) instead rotates the whole field by 90 deg.
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))
    for ax, (u_, v_, title) in zip(
        axes[:2],
        [(Wt, V, "TG-vortex (deviation) velocity  $(\\tilde w, \\tilde v)$"),
         (Wtot, V, "total secondary velocity  $(W_{DNS}+\\tilde w, V)$")],
    ):
        q = ax.quiver(Z[::s, ::s], Y[::s, ::s], u_[::s, ::s], v_[::s, ::s],
                      np.hypot(u_, v_)[::s, ::s], cmap="viridis",
                      pivot="mid", scale=qscale, scale_units="xy",
                      width=0.0035)
        ax.set_xlabel(f"$z/h$   ($\\lambda_3 = {lam}h$, {args.n_lam} periods)")
        ax.set_ylabel("$y/h$")
        ax.set_title(title, fontsize=10)
        plt.colorbar(q, ax=ax, label="speed ($u_\\tau$)")
    ax = axes[2]
    spd = np.hypot(V, Wt)
    strm = ax.streamplot(zrow, Y[:, 0], Wt, V, color=spd, cmap="viridis",
                         density=1.7, linewidth=0.9, arrowsize=0.9)
    ax.set_xlabel(f"$z/h$   ($\\lambda_3 = {lam}h$, {args.n_lam} periods)")
    ax.set_ylabel("$y/h$")
    ax.set_title("streamlines of the deviation velocity", fontsize=10)
    plt.colorbar(strm.lines, ax=ax, label="speed ($u_\\tau$)")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "vectors.png"), dpi=170)
    plt.close(fig)

    # --- Fig 2: streamwise vorticity + stream function ---
    # fluctuation (TG-vortex) vorticity = Omega_x - dWbar/dy
    Wp_arr = profiles["W"].deriv(1).np(y1)
    Om_f = Om - Wp_arr[:, None]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    ext = [0, lam * args.n_lam, -1, 1]
    vmax = np.abs(Om_f).max()
    im = axes[0].imshow(Om_f, origin="lower", aspect="auto", extent=ext,
                        cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[0].set_title(r"fluctuation vorticity $\widetilde\Omega_x=\Omega_x"
                      r"-\mathrm{d}\bar W/\mathrm{d}y$ (TG vortex)")
    axes[0].set_xlabel("$z/h$"); axes[0].set_ylabel("$y/h$")
    plt.colorbar(im, ax=axes[0])
    nlev = 21
    psi_max = np.abs(Ps).max()
    levels = np.linspace(-psi_max, psi_max, nlev)
    cf = axes[1].contourf(Z, Y, Ps, levels=levels, cmap="RdBu_r")
    cs = axes[1].contour(Z, Y, Ps, levels=levels[1:-1], colors="k",
                         linewidths=0.3)
    axes[1].clabel(cs, fontsize=6, fmt="%.2f")
    axes[1].set_title(r"stream function $\psi$ of the vortex part")
    axes[1].set_xlabel("$z/h$"); axes[1].set_ylabel("$y/h$")
    plt.colorbar(cf, ax=axes[1])
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "vorticity_streamfunction.png"), dpi=170)
    plt.close(fig)

    # --- Fig 3: z-mean profile checks ---
    v2 = (f["V"].view(ny, nz) ** 2).mean(dim=1).detach().numpy()
    w2 = (f["Wt"].view(ny, nz) ** 2).mean(dim=1).detach().numpy()
    om_mean = Om.mean(axis=1)
    vv_dns = profiles["vv"].np(y1)
    W_dns = profiles["W"].np(y1)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    ax = axes[0]
    ax.plot(v2, y1, label=r"$\langle \tilde v^2 \rangle_z$")
    ax.plot(w2, y1, "--", label=r"$\langle \tilde w^2 \rangle_z$")
    ax.plot(cfg["beta"] * vv_dns, y1, ":", label=rf"$\beta\,\langle vv\rangle^+$ ($\beta$={cfg['beta']})")
    ax.set_xlabel("energy ($u_\\tau^2$)"); ax.set_ylabel("$y/h$")
    ax.legend(fontsize=8); ax.set_title("vortex energy calibration (L4)")
    ax = axes[1]
    ax.plot(f["Wbar"].view(ny, nz).detach().numpy()[:, 0], y1, "-", label="PINN $\\bar W(y)$ (= DNS, exact)")
    ax.plot(dns["W"], dns["y"], "o", ms=2.5, label="DNS $W^+$ data")
    ax.set_xlabel("$W^+$"); ax.set_ylabel("$y/h$")
    ax.legend(fontsize=8); ax.set_title("mode-1 profile locked to DNS (L3)")
    ax = axes[2]
    ax.plot(om_mean, y1, label=r"$\langle \Omega_x \rangle_z$")
    ax.plot(Wp_arr, y1, ":", label=r"$\mathrm{d}\bar W/\mathrm{d}y$")
    ax.set_xlabel(r"$\Omega_x$"); ax.set_ylabel("$y/h$")
    ax.legend(fontsize=8); ax.set_title("mean streamwise vorticity")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "profiles.png"), dpi=170)
    plt.close(fig)

    # --- Fig 4: spanwise spectra of v~ at selected y ---
    nper = nz // args.n_lam          # points per wavelength
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for yy in (-0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8):
        j = int(np.argmin(np.abs(y1 - yy)))
        vrow = V[j, :].reshape(args.n_lam, nper).mean(axis=0)
        spec = np.abs(np.fft.rfft(vrow)) ** 2
        spec[0] = 0.0
        if spec[1] == 0:
            continue
        spec = spec / spec[1]
        ax.semilogy(np.arange(1, min(len(spec), 17)),
                    spec[1:17], marker="o", ms=3, label=f"$y/h$={yy:+.1f}")
    ax.set_xlabel("harmonic $k$ (period $\\lambda_3/k$)")
    ax.set_ylabel("$E(k)/E(1)$")
    ax.set_title("spanwise spectrum of $\\tilde v$")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "spectra.png"), dpi=170)
    plt.close(fig)

    # --- summary table -----------------------------------------------------
    # roll count from the *fluctuation* vorticity (the mean vorticity sheet
    # dWbar/dy dominates Omega_x near the walls and has no z-structure)
    def count_rolls(om_row):
        sgn = np.sign(om_row)
        return int(np.sum(np.diff(sgn) != 0)) if len(om_row) > 1 else 0
    interior = slice(6, ny - 6)                      # skip wall layers
    jmax = 6 + int(np.argmax(np.abs(Om_f).max(axis=1)[interior]))
    rolls = count_rolls(Om_f[jmax]) // 2

    summary = {
        "config": {k: cfg[k] for k in
                   ("beta", "lam", "m", "adam", "lbfgs", "seed")},
        "diagnostics": diag,
        "vortex_scales": {
            "max_abs_V_utau": float(np.abs(V).max()),
            "max_abs_Wt_utau": float(np.abs(Wt).max()),
            "max_abs_Omega_total": float(np.abs(Om).max()),
            "max_abs_Omega_fluct": float(np.abs(Om_f).max()),
            "rms_R": float(np.sqrt((R ** 2).mean())),
            "vortex_pairs_per_lambda": rolls,
            "V_at_strongest_row_utau": float(np.abs(V[jmax]).max()),
            "y_of_strongest_vorticity": float(y1[jmax]),
        },
    }
    # v2 additions (append-only: every legacy key above is preserved)
    st = getattr(net, "structure", None) or {}
    if st:
        summary["config"].update({
            "structure_hash": getattr(net, "structure_hash", ""),
            "param_count": int(getattr(net, "param_count", 0)),
            "activation": st["trunk"]["activation"],
            "skip": st["trunk"]["skip"],
            "heads_mode": st["heads"]["mode"],
            "wall_type": st["wall"]["type"],
        })
    _trapz = getattr(np, "trapezoid", None)
    if _trapz is None:                      # numpy < 2.0
        _trapz = np.trapz
    e_int = float(_trapz(v2, y1))
    e_tgt = float(_trapz(cfg["beta"] * vv_dns, y1))
    summary["vortex_scales"].update({
        "roll_sign_changes": int(count_rolls(Om_f[jmax])),
        "energy_integral_ratio": e_int / e_tgt if e_tgt > 0 else float("nan"),
    })
    with open(os.path.join(outdir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    np.savez(os.path.join(outdir, "fields.npz"), Y=Y, Z=Z, V=V, Wt=Wt,
             Wtot=Wtot, Om=Om, Omf=Om_f, phi=Ps, R=R, y1=y1, lam=lam)
    print(json.dumps(summary["vortex_scales"], indent=2))
    print("figures saved to", figdir)


if __name__ == "__main__":
    main()
