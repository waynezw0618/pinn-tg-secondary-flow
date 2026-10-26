"""Unified checkpoint loading for legacy (pre-v2) and structured TGNet models.

Two facts make this module necessary:

* pre-v2 checkpoints store ``cfg`` only (no structure) and a
  ``state_dict`` whose keys are ``mlp.0.weight`` ... ``mlp.{2d}.weight``;
* v2 checkpoints may use ``trunk.*`` / ``heads.*`` (or ``basis.*``).

The loader never guesses silently: if a key remap is required it returns the
list of applied remappings to the caller so training logs and the audit trail
can record it.
"""
from __future__ import annotations

import json
import os

import torch

from .pinn import TGNet
from .structure import (is_legacy_equivalent, legacy_structure,
                        normalize_structure, validate_structure)


def structure_from_ckpt(ckpt: dict) -> dict:
    """Resolve the structure of a checkpoint (v2 field or legacy cfg)."""
    s = ckpt.get("structure")
    if s:
        if isinstance(s, str):
            s = json.loads(s)
        return normalize_structure(s)
    cfg = ckpt.get("cfg", {})
    missing = [k for k in ("lam", "m", "width", "depth") if k not in cfg]
    if missing:
        raise KeyError(f"checkpoint has neither 'structure' nor legacy cfg "
                       f"keys {missing}; cfg keys = {sorted(cfg)}")
    return legacy_structure(cfg["lam"], cfg["m"], cfg["width"], cfg["depth"],
                            float(cfg.get("out_scale", 0.05)))


def load_net_from_ckpt(ckpt: dict, device: str = "cpu"):
    """Build the network described by ``ckpt`` and load its weights."""
    s = structure_from_ckpt(ckpt)
    errs = validate_structure(s)
    if errs:
        raise ValueError(f"invalid structure in checkpoint: {errs}")
    net = TGNet(structure=s)
    sd = ckpt["state_dict"]
    try:
        net.load_state_dict(sd, strict=True)
        remaps: list[str] = []
    except RuntimeError:
        sd2, remaps = remap_legacy_state_dict(sd, s)
        net.load_state_dict(sd2, strict=True)
    net.to(device)
    net.eval()
    net.load_report = {"remapped": remaps, "legacy": net.is_legacy}
    return net


def load_run_net(run_dir: str, device: str = "cpu"):
    """Load ``<run_dir>/model.pt`` -> ``(net, cfg, structure)``."""
    path = os.path.join(run_dir, "model.pt")
    ckpt = torch.load(path, weights_only=False, map_location=device)
    net = load_net_from_ckpt(ckpt, device)
    return net, ckpt.get("cfg", {}), net.structure


def remap_legacy_state_dict(sd: dict, structure: dict) -> tuple[dict, list[str]]:
    """Map a legacy ``mlp.*`` state dict onto a structured network.

    Only the ``shared`` head case is remappable key-for-key; anything else is
    reported as an explicit error rather than a silent partial load.
    """
    s = normalize_structure(structure)
    if is_legacy_equivalent(s):
        # Same structure but key mismatch should not happen; surface it.
        return sd, []

    keys = list(sd)
    if not any(k.startswith("mlp.") for k in keys):
        return sd, []

    if s["heads"]["mode"] != "shared":
        raise RuntimeError(
            "cannot remap a legacy 'mlp.*' state_dict onto heads.mode="
            f"{s['heads']['mode']}; retrain this structure instead")

    width, depth = s["trunk"]["width"], s["trunk"]["depth"]
    out = {}
    notes: list[str] = []
    out["trunk.inp.weight"] = sd["mlp.0.weight"]
    out["trunk.inp.bias"] = sd["mlp.0.bias"]
    notes.append("mlp.0 -> trunk.inp")
    for i in range(depth - 1):
        src = f"mlp.{2 * (i + 1)}"
        dst = f"trunk.blocks.{i}"
        out[f"{dst}.weight"] = sd[f"{src}.weight"]
        out[f"{dst}.bias"] = sd[f"{src}.bias"]
        notes.append(f"{src} -> {dst}")
    last = f"mlp.{2 * depth}"
    out["heads.weight"] = sd[f"{last}.weight"]
    out["heads.bias"] = sd[f"{last}.bias"]
    notes.append(f"{last} -> heads")
    return out, notes


def save_checkpoint(path: str, net: TGNet, cfg: dict, **extra) -> None:
    """Persist a v2 checkpoint (structure + provenance fields included)."""
    payload = {
        "state_dict": net.state_dict(),
        "cfg": cfg,
        "schema_version": net.structure["schema_version"],
        "structure": net.structure,
        "structure_hash": net.structure_hash,
        "param_count": net.param_count,
    }
    payload.update(extra)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(payload, path)
