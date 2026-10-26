"""Training driver for the 1D PINN-RANS-ASM solver.

Usage (project root):
    python -m pinn_tg.train_1d --ro 30 --adam 3000 --lbfgs 100 --name rasm_1d_ro30

Saves to outputs/runs/<name>/:
    config.json      hyperparameters
    history.json     per-step loss log
    model.pt         trained network weights
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from .pinn_1d import MLP1D, PINN1DConfig, RANSProblem1D


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ro", type=float, default=30.0,
                    help="Rotation number Ro_tau")
    ap.add_argument("--re", type=float, default=180.0,
                    help="Friction Reynolds number Re_tau")
    ap.add_argument("--width", type=int, default=64, help="MLP width")
    ap.add_argument("--depth", type=int, default=4, help="MLP depth")
    ap.add_argument("--n-col", type=int, default=200,
                    help="Number of collocation points (uniform in yhat)")
    ap.add_argument("--adam", type=int, default=3000, help="Adam steps")
    ap.add_argument("--lbfgs", type=int, default=100, help="L-BFGS max_iter")
    ap.add_argument("--lr", type=float, default=1e-3, help="Adam initial lr")
    ap.add_argument("--w-pde", type=float, default=1.0)
    ap.add_argument("--w-bc", type=float, default=50.0)
    ap.add_argument("--w-real", type=float, default=10.0)
    ap.add_argument("--w-data", type=float, default=100.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", type=str, default="")
    return ap.parse_args()


def main():
    args = parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    name = args.name or f"rasm_1d_ro{args.ro:g}_seed{args.seed}"
    outdir = os.path.join(root, "outputs", "runs", name)
    os.makedirs(outdir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = PINN1DConfig(
        Re_tau=args.re,
        Ro_tau=args.ro,
        width=args.width,
        depth=args.depth,
        w_pde=args.w_pde,
        w_bc=args.w_bc,
        w_real=args.w_real,
        w_data=args.w_data,
    )
    net = MLP1D(cfg.width, cfg.depth)
    prob = RANSProblem1D(net, cfg)

    cfg_dump = vars(args).copy()
    cfg_dump.update(dict(
        n_params=sum(p.numel() for p in net.parameters()),
        Omega_outer=prob.Omega_outer,
        nu_outer=prob.nu_outer,
    ))
    with open(os.path.join(outdir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg_dump, f, indent=2)

    t0 = time.time()
    hist = []
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    # 固定的 yhat 配点 + 边界点
    yhat_col = torch.linspace(-1, 1, args.n_col,
                              dtype=torch.float32).unsqueeze(-1)
    yhat_col.requires_grad_(True)
    yhat_walls = torch.tensor([[-1.0], [1.0]],
                              dtype=torch.float32, requires_grad=True)

    # DNS 锚点（破 PDE 齐次平凡解）。Roτ=30 才有 DNS。
    # 文件按 wall units，y/h∈[-1,1]；直接当 PINN 的 yhat 表驱动。
    yhat_data = None
    U_dns = None
    W_dns = None
    dns_path = os.path.join(root, "data", "Retau180Ro30.txt")
    if os.path.exists(dns_path) and abs(args.ro - 30.0) < 1e-6:
        from .data import load_dns
        dns, _ = load_dns(dns_path)
        # DNS y 是浮点 array，归一到 [-1,1] 直接当 yhat；100 个采样点够
        idx = np.linspace(0, len(dns["y"]) - 1, 100).astype(int)
        yhat_data = torch.tensor(
            dns["y"][idx], dtype=torch.float32).unsqueeze(-1)
        U_dns = torch.tensor(
            dns["U"][idx], dtype=torch.float32).unsqueeze(-1)
        W_dns = torch.tensor(
            dns["W"][idx], dtype=torch.float32).unsqueeze(-1)
        print(f"DNS data anchor: {len(idx)} points "
              f"y range [{yhat_data.min().item():.3f}, {yhat_data.max().item():.3f}]")

    def step_loss():
        loss_dict = prob.loss(
            yhat_col, yhat_walls,
            yhat_data=yhat_data, U_dns=U_dns, W_dns=W_dns)
        return (loss_dict["total"],
                dict(L_pde=loss_dict["pde"].detach().item(),
                     L_bc=loss_dict["bc"].detach().item(),
                     L_real=loss_dict["real"].detach().item()
                     if torch.is_tensor(loss_dict["real"]) else float(loss_dict["real"]),
                     L_data=loss_dict["data"].detach().item()
                     if torch.is_tensor(loss_dict["data"]) else 0.0))

    print(f"training '{name}': Adam {args.adam} steps ...")
    last_good = 0
    for step in range(1, args.adam + 1):
        # LR step decay: full lr first 60%, decay to 0.2x after
        lr = args.lr if step <= 0.6 * args.adam else args.lr * 0.2
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad()
        loss, diag = step_loss()
        # NaN/inf 保护：跳过 bad step
        if not np.isfinite(float(loss.detach())):
            if step == 1 or step % 200 == 0:
                print(f"  step {step:5d}  SKIP loss={float(loss.detach()):.3e}")
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        opt.step()
        last_good = step
        if step == 1 or step % 200 == 0:
            hist.append(dict(step=step, loss=float(loss.detach()), **diag))
            print(f"  step {step:5d}  loss {float(loss.detach()):.4e}  "
                  f"pde {diag['L_pde']:.3e}  bc {diag['L_bc']:.3e}  "
                  f"real {diag['L_real']:.3e}  data {diag['L_data']:.3e}  "
                  f"({time.time()-t0:.0f}s)")

    print(f"L-BFGS refinement ({args.lbfgs} iters, last_good Adam={last_good}) ...")
    # 保存 Adam 末态，L-BFGS 出 NaN 时回滚
    adam_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    opt2 = torch.optim.LBFGS(net.parameters(), lr=0.5,
                             max_iter=args.lbfgs, history_size=50,
                             line_search_fn="strong_wolfe")

    def closure():
        opt2.zero_grad()
        loss_dict = prob.loss(
            yhat_col, yhat_walls,
            yhat_data=yhat_data, U_dns=U_dns, W_dns=W_dns)
        L = loss_dict["total"]
        if not torch.isfinite(L):
            # 损失非有限：梯度置零、返回有限大值，让 line search 自然回退
            for p in net.parameters():
                p.grad = torch.zeros_like(p)
            return torch.tensor(1e6, dtype=L.dtype, device=L.device)
        L.backward()
        return L

    try:
        opt2.step(closure)
    except Exception as e:  # noqa: BLE001
        print("LBFGS stopped:", e)
    loss, diag = step_loss()
    if not np.isfinite(float(loss.detach())):
        print("L-BFGS produced NaN — rolling back to Adam state.")
        net.load_state_dict(adam_state)
        loss, diag = step_loss()
    hist.append(dict(step=args.adam + args.lbfgs, loss=float(loss), **diag))
    print(f"final: loss {float(loss):.4e}  pde {diag['L_pde']:.3e}  "
          f"bc {diag['L_bc']:.3e}  real {diag['L_real']:.3e}  "
          f"data {diag.get('L_data', 0.0):.3e}  "
          f"time {time.time()-t0:.0f}s")

    torch.save({
        "state_dict": net.state_dict(),
        "cfg": cfg_dump,
        "Ro_tau": args.ro,
        "Re_tau": args.re,
    }, os.path.join(outdir, "model.pt"))
    with open(os.path.join(outdir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(hist, f, indent=1)
    print(f"saved: {outdir}")


if __name__ == "__main__":
    main()