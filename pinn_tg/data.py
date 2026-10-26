"""DNS data parsing and differentiable Chebyshev profile fits.

Data file: data/Retau180Ro30.txt (Yang & Wang 2018, Case C)
Columns (first 10): y/h, U+, W+, <uu>+, <vv>+, <ww>+, <uv>+, <uw>+, <vw>+, TKE+
All quantities in wall units (u_tau, nu); y/h in [-1, 1]; Re_tau=180, Ro_tau=30.

Outer (PINN) units used downstream: lengths by h, velocities by u_tau,
hence nu_outer = 1/Re_tau and nu_t_outer = nu_t_plus / Re_tau.
"""
from __future__ import annotations

import numpy as np
import torch
from numpy.polynomial import chebyshev as Ccheb

RE_TAU = 180.0
COLNAMES = ["y", "U", "W", "uu", "vv", "ww", "uv", "uw", "vw", "TKE"]


def load_dns(path):
    """Parse the %-commented DNS profile file, return dict of 1-D arrays."""
    rows = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("%"):
                continue
            parts = s.split()
            if len(parts) < 10:
                continue
            try:
                rows.append([float(v) for v in parts[:10]])
            except ValueError:
                continue
    arr = np.asarray(rows, dtype=float)
    arr = arr[np.argsort(arr[:, 0])]
    if arr[0, 0] < -1.0001 or arr[-1, 0] > 1.0001:
        raise ValueError("y/h out of [-1, 1]")
    data = {name: arr[:, i].copy() for i, name in enumerate(COLNAMES)}
    return data, arr.shape[0]


def load_dns_budget(path):
    """Load the full Yang & Wang (2018) budget file (58 columns) with names.

    The file header (a '%'-commented line beginning with 'y/h') lists every
    column.  Budget columns follow the sign convention
        res_ij = produc_ij + rot_ij + p-strain_ij + dissip_ij + t-diff_ij
                 + p-diff_ij + v-diff_ij
    so *dissipation is stored negative* (it is a sink).

    Returns:
        (dict of named columns, plus)
        'eps' = -(dissip_uu + dissip_vv + dissip_ww)/2   [wall units, positive]
                i.e. the TKE dissipation rate eps+ = 1/2 eps_ii+.
    """
    names = None
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("%"):
                body = s.lstrip("%").strip()
                if body.startswith("y/h") and names is None:
                    names = body.split()
                continue
            parts = s.split()
            if parts:
                rows.append([float(v) for v in parts])
    arr = np.asarray(rows, dtype=float)
    arr = arr[np.argsort(arr[:, 0])]
    if names is None or len(names) != arr.shape[1]:
        # fall back to positional names
        names = [f"c{i}" for i in range(arr.shape[1])]
    out = {nm: arr[:, i].copy() for i, nm in enumerate(names)}
    # TKE dissipation = 1/2 trace of the dissipation tensor; stored as -eps.
    # Header names carry a trailing '+' (e.g. 'dissip_uu+').
    dk = [f"dissip_{c}+" for c in ("uu", "vv", "ww")]
    if all(k in out for k in dk):
        out["eps"] = -0.5 * sum(out[k] for k in dk)
        out["eps_source"] = "-(dissip_uu+ + dissip_vv+ + dissip_ww+)/2"
    return out, arr.shape[1]


class ChebProfile:
    """Smooth torch-differentiable profile fit on y in [-1, 1].

    Evaluates the Chebyshev series with a differentiable three-term recurrence,
    so autograd can take derivatives w.r.t. y through the profile.
    """

    def __init__(self, y, v, deg=24):
        ok = np.isfinite(v)
        self.cheb = Ccheb.Chebyshev.fit(y[ok], v[ok], deg, domain=[-1.0, 1.0])
        self.deg = deg
        self._set_coeffs(self.cheb.coef)

    def _set_coeffs(self, coef):
        c = np.asarray(coef, dtype=np.float64)
        if len(c) < 2:
            c = np.concatenate([c, np.zeros(2 - len(c))])
        self.coeffs = torch.tensor(c, dtype=torch.float32)

    def deriv(self, n=1):
        """Analytic n-th derivative as a new ChebProfile."""
        cp = ChebProfile.__new__(ChebProfile)
        cp.cheb = self.cheb.deriv(n)
        cp.deg = cp.cheb.degree()
        cp._set_coeffs(cp.cheb.coef)
        return cp

    def __call__(self, y):
        c = self.coeffs.to(y.dtype)
        T0 = torch.ones_like(y)
        T1 = y
        out = c[0] * T0 + c[1] * T1
        for k in range(2, len(c)):
            T2 = 2.0 * y * T1 - T0
            out = out + c[k] * T2
            T0, T1 = T1, T2
        return out

    def np(self, yg):
        return self.cheb(yg)


def build_profiles(dns, deg_U=36, deg_W=24, deg_vv=24, deg_nut=16):
    """Fit all profiles needed by the PINN.

    nu_t^+(y) = -<uv>+ / (dU+/dy+), capped to [0, 40] wall units, then lightly
    smoothed and fitted; converted to outer units (divide by Re_tau).
    """
    y = dns["y"]
    U = ChebProfile(y, dns["U"], deg=deg_U)
    W = ChebProfile(y, dns["W"], deg=deg_W)
    vv = ChebProfile(y, dns["vv"], deg=deg_vv)
    uv = ChebProfile(y, dns["uv"], deg=24)

    yg = np.linspace(-1.0, 1.0, 721)
    dUdy_out = U.cheb.deriv()(yg)              # d(U+)/d(y/h)
    uvp = uv.cheb(yg)
    nutp = -uvp * RE_TAU / dUdy_out            # wall units
    nutp = np.clip(nutp, 0.0, 40.0)
    from scipy.signal import savgol_filter
    nutp_s = savgol_filter(nutp, 41, 3, mode="nearest")
    nut = ChebProfile(yg, nutp_s / RE_TAU, deg=deg_nut)

    return {
        "U": U, "W": W, "vv": vv, "uv": uv, "nut": nut,
        "vv_max": float(np.max(dns["vv"])),
        "nutp_max": float(np.max(nutp_s)),
    }


def load_asm_profiles(npz_path, deg_W=24, deg_nut=16, deg_vv=24,
                      nut_cap=40.0, source="asm"):
    """Build 2D-PINN-compatible profiles from a 1D PINN-RANS-ASM run.

    Reads fields.npz (written by pinn_tg.postprocess_1d) and fits the same
    ChebProfile interfaces used by TGProblem:
        W   <- npz['W']    (mean spanwise velocity, wall units)
        nut <- npz['nu_t'] (eddy viscosity, converted to outer units)
        vv  <- npz['vv']   (ASM <vv> component)
        vv_max <- max(npz['vv'])

    Args:
        npz_path: path to fields.npz
        nut_cap: cap nu_t in wall units before fitting (mirrors build_profiles)
        source: bookkeeping tag stored in the returned dict

    Returns:
        dict compatible with build_profiles output (keys used by TGProblem:
        W, nut, vv_max, plus extras for logging).
    """
    d = np.load(npz_path)
    y = np.asarray(d["y"], dtype=np.float64)
    re_tau = float(d["Re_tau"]) if "Re_tau" in d else RE_TAU

    W_arr = np.asarray(d["W"], dtype=np.float64)
    vv_arr = np.asarray(d["vv"], dtype=np.float64)
    nut_arr = np.asarray(d["nu_t"], dtype=np.float64)

    # nu_t 在 npz 中是外尺度单位；转 wall units 便于 cap，再转回
    nutp = np.clip(nut_arr * re_tau, 0.0, nut_cap)
    nut = ChebProfile(y, nutp / re_tau, deg=deg_nut)
    W = ChebProfile(y, W_arr, deg=deg_W)
    vv = ChebProfile(y, vv_arr, deg=deg_vv)

    return {
        "W": W, "vv": vv, "nut": nut,
        "vv_max": float(np.max(vv_arr)),
        "nutp_max": float(np.max(nutp)),
        "source": source, "npz_path": str(npz_path),
    }
