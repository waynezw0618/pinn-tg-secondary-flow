"""Training driver for the TG-vortex PINN.

Run (project root):
    python -m pinn_tg.train --beta 0.05 --lam 1.5 --adam 4000 --lbfgs 200
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from .data import load_dns, build_profiles
from .pinn import TGNet, TGProblem
from .structure import (legacy_structure, normalize_structure,
                        validate_structure)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, default=0.05,
                    help="vortex energy fraction: <v~^2> = beta*<vv>_DNS")
    ap.add_argument("--lam", type=float, default=1.5,
                    help="spanwise wavelength lambda_3 in half-channel heights")
    ap.add_argument("--m", type=int, default=8, help="Fourier harmonics in z")
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--ny", type=int, default=201)
    ap.add_argument("--nz", type=int, default=64)
    ap.add_argument("--adam", type=int, default=4000)
    ap.add_argument("--lbfgs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--w-pde", type=float, default=1.0)
    ap.add_argument("--w-energy", type=float, default=1.0)
    ap.add_argument("--no-nut", action="store_true")
    ap.add_argument("--no-aniso", action="store_true")
    ap.add_argument("--data", type=str, default="Retau180Ro30.txt",
                    help="DNS data file name under data/")
    ap.add_argument("--asm-npz", type=str, default="",
                    help="path to 1D PINN-RANS-ASM fields.npz; if set, "
                         "replaces DNS profiles with ASM output")
    ap.add_argument("--ro", type=float, default=30.0,
                    help="rotation number Ro_tau of the case (bookkeeping)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", type=str, default="")
    ap.add_argument("--structure", type=str, default="",
                    help="structure JSON string or path to a JSON file "
                         "(see pinn_tg.structure); empty = legacy structure")
    ap.add_argument("--init-from", type=str, default="",
                    help="run name or .pt path for a warm start")
    ap.add_argument("--time-budget", type=float, default=0.0,
                    help="wall-clock budget in seconds; <=0 disables")
    ap.add_argument("--early-stop", type=int, default=0,
                    help="Adam patience on loss stagnation; 0 disables")
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--force", action="store_true",
                    help="allow overwriting an existing agent_* run")
    return ap.parse_args()


def resolve_structure(args):
    """Structure is authoritative; legacy args are used when it is absent."""
    if not args.structure:
        return legacy_structure(args.lam, args.m, args.width, args.depth)
    raw = args.structure
    if os.path.exists(raw):
        with open(raw, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
    else:
        cfg = json.loads(raw)
    cfg = normalize_structure(cfg)
    errs = validate_structure(cfg)
    if errs:
        raise SystemExit("invalid --structure: " + "; ".join(errs))
    return cfg


def warm_start(net, spec, root):
    """Load matching tensors from a previous run/checkpoint (shape-safe)."""
    import torch as _t
    path = spec
    if not os.path.isfile(path):
        cand = os.path.join(root, "outputs", "runs", spec, "model.pt")
        if os.path.isfile(cand):
            path = cand
        else:
            raise SystemExit(f"--init-from: cannot find {spec!r}")
    ckpt = _t.load(path, weights_only=False, map_location="cpu")
    src = ckpt["state_dict"]
    dst = net.state_dict()
    loaded, skipped = [], []
    for k, v in src.items():
        if k in dst and dst[k].shape == v.shape:
            dst[k] = v
            loaded.append(k)
        else:
            skipped.append(k)
    net.load_state_dict(dst)
    print(f"warm start: {len(loaded)} tensors from {path}, "
          f"{len(skipped)} skipped (shape/key mismatch)")
    return {"path": path, "loaded": len(loaded), "skipped": skipped[:20]}


def main():
    args = parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    structure = resolve_structure(args)
    # structure is authoritative for the network shape
    args.lam = float(structure["lam"])
    args.m = int(structure["m"])
    args.width = int(structure["trunk"]["width"])
    args.depth = int(structure["trunk"]["depth"])
    name = args.name or f"beta{args.beta}_lam{args.lam}_seed{args.seed}"
    outdir = os.path.join(root, "outputs", "runs", name)
    if (name.startswith("agent_") and not args.force
            and os.path.exists(os.path.join(outdir, "model.pt"))):
        raise SystemExit(f"refusing to overwrite agent run {name!r} "
                         f"(pass --force to override)")
    os.makedirs(outdir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.asm_npz:
        # 1D PINN-RANS-ASM 链路：用 ASM 输出替换 DNS 剖面
        from .data import load_asm_profiles
        asm_npz = (args.asm_npz if os.path.isabs(args.asm_npz)
                   else os.path.join(root, args.asm_npz))
        profiles = load_asm_profiles(asm_npz)
        profiles["U"] = None  # TGProblem 不用 U，占位保持 dict 结构
        npts = -1
        print(f"profiles: ASM npz {asm_npz}")
    else:
        dns, npts = load_dns(os.path.join(root, "data", args.data))
        profiles = build_profiles(dns)
    prob = TGProblem(profiles, lam=args.lam, ny=args.ny, nz=args.nz,
                     beta=args.beta, use_nut=not args.no_nut,
                     use_aniso=not args.no_aniso)
    net = TGNet(structure=structure)
    if args.init_from:
        warm = warm_start(net, args.init_from, root)
    else:
        warm = None
    print(f"structure: {net.describe()}  params={net.param_count} "
          f"hash={net.structure_hash}")

    cfg = vars(args).copy()
    cfg.update(dict(n_dns_points=npts, vv_max=profiles["vv_max"],
                    nutp_max=profiles["nutp_max"], U_s2=prob.U_s2,
                    structure=structure, structure_hash=net.structure_hash,
                    param_count=net.param_count, warm_start=warm))
    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    t0 = time.time()
    hist = []
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    def total_loss():
        Lp, Le, diag, _ = prob.losses(net)
        return (args.w_pde * Lp + args.w_energy * Le), diag

    print(f"training '{name}': Adam {args.adam} steps ...")
    best = float("inf")
    last_improve = 0
    adam_done = args.adam
    for step in range(1, args.adam + 1):
        if args.time_budget > 0 and (time.time() - t0) > args.time_budget:
            print(f"time budget {args.time_budget:.0f}s reached at step {step}")
            adam_done = step - 1
            break
        # simple step decay
        lr = args.lr if step <= 0.6 * args.adam else args.lr * 0.2
        for g in opt.param_groups:
            g["lr"] = lr
        loss, diag = total_loss()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        if not np.isfinite(float(loss.detach())):
            print("NaN loss at step", step, "- abort")
            break
        cur = float(loss.detach())
        if cur < best * (1.0 - 1e-4):
            best, last_improve = cur, step
        if step == 1 or step % 200 == 0:
            hist.append(dict(step=step, loss=cur, **diag))
            print(f"  step {step:5d}  loss {cur:.4e}  "
                  f"pde {diag['L_pde']:.3e}  energy {diag['L_energy']:.3e}  "
                  f"maxV {diag['max_V']:.3f}  maxWt {diag['max_Wt']:.3f}  "
                  f"({time.time()-t0:.0f}s)")
        if args.early_stop and step - last_improve > args.early_stop:
            print(f"early stop at step {step}: no gain for "
                  f"{args.early_stop} steps (best {best:.4e})")
            adam_done = step
            break

    skipped_lbfgs = False
    if args.time_budget > 0 and (time.time() - t0) > args.time_budget:
        print("time budget exhausted: skipping L-BFGS")
        skipped_lbfgs = True
    if not skipped_lbfgs:
        print("L-BFGS refinement ...")
        opt2 = torch.optim.LBFGS(net.parameters(), lr=0.5,
                                 max_iter=args.lbfgs, history_size=60,
                                 line_search_fn="strong_wolfe")

        def closure():
            loss, diag = total_loss()
            opt2.zero_grad()
            loss.backward()
            return loss

        try:
            opt2.step(closure)
        except Exception as e:  # noqa: BLE001
            print("LBFGS stopped:", e)
    loss, diag = total_loss()
    hist.append(dict(step=adam_done + args.lbfgs, loss=float(loss.detach()),
                     **diag))
    print(f"final: loss {float(loss):.4e}  pde {diag['L_pde']:.3e}  "
          f"energy {diag['L_energy']:.3e}  maxV {diag['max_V']:.3f}  "
          f"maxWt {diag['max_Wt']:.3f}  time {time.time()-t0:.0f}s")

    from .checkpoint import save_checkpoint
    save_checkpoint(os.path.join(outdir, "model.pt"), net, cfg,
                    adam_steps_done=adam_done, lbfgs_skipped=skipped_lbfgs,
                    wall_time_s=round(time.time() - t0, 1),
                    created_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    with open(os.path.join(outdir, "history.json"), "w") as f:
        json.dump(hist, f, indent=1)
    print("saved:", outdir)


if __name__ == "__main__":
    main()
