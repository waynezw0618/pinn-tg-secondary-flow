"""Declarative network-structure schema for the TG-vortex PINN (v2).

Why this module exists
----------------------
The original ``TGNet`` exposed only four scalars (``lam, m, width, depth``).
Structured discovery (Psi-NN) and LLM-driven proposals need a *serialisable*
description of the network so a candidate can be proposed, validated,
trained, compared and reproduced without touching code.

Design contract
---------------
1. ``legacy_structure(...)`` reproduces **exactly** the network that the
   pre-v2 code built (shared MLP trunk, tanh, ``(1-y^2)^2`` wall factor).
   For that structure ``TGNet`` keeps the module name ``mlp`` and the same
   layer order, so every historical ``model.pt`` still loads with
   ``strict=True`` and ``forward`` stays bit-identical.
2. Any other structure uses ``trunk`` + ``heads`` and is self-describing
   (``schema_version`` is stored in the checkpoint).
3. A structure is a plain ``dict`` -> JSON, so it can be produced by an LLM,
   diffed, hashed and stored in SQLite.

Honest note on "compression"
----------------------------
``shared``, ``grouped`` and ``per_mode`` have **identical** parameter counts:
they only differ in how the final linear map is factorised. Real parameter
savings come from ``tied_grouped`` (shared low-rank basis across groups),
shrinking ``trunk.width``, or dropping modes via ``mask.active_modes``.
Do not claim compression for a mere re-grouping.
"""
from __future__ import annotations

import hashlib
import json
import random
from typing import Iterable

SCHEMA_VERSION = 2

ACTIVATIONS = ("tanh", "sin", "gelu")
SKIPS = ("none", "residual")
NORMS = ("none", "layernorm")
HEAD_MODES = ("shared", "grouped", "per_mode", "tied_grouped")
WALL_TYPES = ("power2", "power2_plus_lag", "exp_blend")

MAX_PARAMS = 12000
MAX_M = 32


DEFAULT_STRUCTURE: dict = {
    "schema_version": SCHEMA_VERSION,
    "lam": 2.0,
    "m": 6,
    "trunk": {"width": 48, "depth": 3, "activation": "tanh",
              "skip": "none", "norm": "none"},
    "heads": {"mode": "shared", "group_size": 1, "groups": None,
              "tie_weights": False, "rank": 0, "out_scale": 0.05},
    "wall": {"type": "power2", "params": {}},
    "mask": {"active_modes": None},
    "init": {"scheme": "xavier_normal", "seed": 0},
    "freeze": {"trunk": False, "heads": False},
}


# ----------------------------------------------------------------------------
def legacy_structure(lam: float, m: int, width: int, depth: int,
                     out_scale: float = 0.05) -> dict:
    """Structure dict equivalent to the pre-v2 ``TGNet`` construction."""
    cfg = json.loads(json.dumps(DEFAULT_STRUCTURE))
    cfg["lam"] = float(lam)
    cfg["m"] = int(m)
    cfg["trunk"]["width"] = int(width)
    cfg["trunk"]["depth"] = int(depth)
    cfg["heads"]["out_scale"] = float(out_scale)
    cfg["mask"]["active_modes"] = list(range(1, int(m) + 1))
    return normalize_structure(cfg)


def _default_groups(m: int, group_size: int) -> list[list[int]]:
    """Contiguous equal groups of **1-based** mode indices."""
    g = max(1, int(group_size))
    return [list(range(i, min(i + g, m + 1))) for i in range(1, m + 1, g)]


def normalize_structure(cfg: dict | None) -> dict:
    """Fill defaults, coerce types, resolve ``groups``. Idempotent."""
    out = json.loads(json.dumps(DEFAULT_STRUCTURE))
    if cfg:
        _deep_update(out, cfg)
    out["schema_version"] = SCHEMA_VERSION
    out["lam"] = float(out["lam"])
    out["m"] = int(out["m"])
    t = out["trunk"]
    t["width"] = int(t["width"])
    t["depth"] = int(t["depth"])
    t["activation"] = str(t["activation"]).lower()
    t["skip"] = str(t["skip"]).lower()
    t["norm"] = str(t["norm"]).lower()
    h = out["heads"]
    h["mode"] = str(h["mode"]).lower()
    h["group_size"] = int(h.get("group_size") or 1)
    h["tie_weights"] = bool(h.get("tie_weights", False))
    h["rank"] = int(h.get("rank") or 0)
    h["out_scale"] = float(h["out_scale"])
    out["wall"]["type"] = str(out["wall"]["type"]).lower()
    if not isinstance(out["wall"].get("params"), dict):
        out["wall"]["params"] = {}

    m = out["m"]
    mode = h["mode"]
    grp = h.get("groups")
    if mode in ("shared",):
        h["groups"] = [list(range(1, m + 1))]
    elif mode == "per_mode":
        h["groups"] = [[i] for i in range(1, m + 1)]
    else:  # grouped / tied_grouped
        if grp:
            # groups are stored 1-based; normalization must be idempotent
            h["groups"] = [sorted(int(i) for i in g) for g in grp]
        else:
            h["groups"] = _default_groups(m, h["group_size"])

    am = out["mask"].get("active_modes")
    if am is None:
        out["mask"]["active_modes"] = list(range(1, m + 1))
    else:
        out["mask"]["active_modes"] = sorted({int(v) for v in am})
    return out


def _deep_update(base: dict, patch: dict) -> None:
    for k, v in patch.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_update(base[k], v)
        else:
            base[k] = v


# ----------------------------------------------------------------------------
def validate_structure(cfg: dict) -> list[str]:
    """Return a list of error codes; empty list means valid."""
    errs: list[str] = []
    c = normalize_structure(cfg)
    m = c["m"]
    t, h, w = c["trunk"], c["heads"], c["wall"]

    if not (1 <= m <= MAX_M):
        errs.append(f"E_M_RANGE: m={m} not in [1,{MAX_M}]")
    if not (c["lam"] > 0):
        errs.append("E_LAM: lam must be > 0")
    if not (4 <= t["width"] <= 512):
        errs.append(f"E_WIDTH_RANGE: width={t['width']} not in [4,512]")
    if not (1 <= t["depth"] <= 8):
        errs.append(f"E_DEPTH_RANGE: depth={t['depth']} not in [1,8]")
    if t["activation"] not in ACTIVATIONS:
        errs.append(f"E_ACT: unknown activation {t['activation']!r}")
    if t["skip"] not in SKIPS:
        errs.append(f"E_SKIP: unknown skip {t['skip']!r}")
    if t["norm"] not in NORMS:
        errs.append(f"E_NORM: unknown norm {t['norm']!r}")
    if h["mode"] not in HEAD_MODES:
        errs.append(f"E_HEAD_MODE: unknown mode {h['mode']!r}")
    if w["type"] not in WALL_TYPES:
        errs.append(f"E_WALL: unknown wall type {w['type']!r}")
    if h["mode"] == "grouped" and m % max(1, h["group_size"]) != 0:
        errs.append(f"E_GROUP_SIZE: group_size={h['group_size']} "
                    f"does not divide m={m}")

    groups = h.get("groups") or []
    if h["mode"] != "shared":
        flat = [i for g in groups for i in g]
        if sorted(flat) != list(range(1, m + 1)):
            errs.append("E_GROUPS: groups must partition modes 1..m exactly")
        if any(len(g) == 0 for g in groups):
            errs.append("E_GROUPS_EMPTY: empty group")
        if any(i < 1 or i > m for i in flat):
            errs.append(f"E_GROUPS_RANGE: mode indices must be within 1..{m}")
        if h["mode"] == "tied_grouped":
            sizes = {len(g) for g in groups}
            if len(sizes) > 1:
                errs.append("E_GROUPS_UNEQUAL: tied_grouped needs equal "
                            "group sizes (low-rank basis is shared)")

    am = c["mask"]["active_modes"]
    if not am:
        errs.append("E_MODES_EMPTY: active_modes is empty")
    elif am[-1] > m or am[0] < 1:
        errs.append(f"E_MODES_RANGE: active_modes must be within 1..{m}")

    try:
        n = param_count(c)
        if n > MAX_PARAMS:
            errs.append(f"E_PARAM_BUDGET: {n} > {MAX_PARAMS}")
    except Exception as exc:  # noqa: BLE001
        errs.append(f"E_PARAM_COMPUTE: {exc}")
    return errs


def is_legacy_equivalent(cfg: dict) -> bool:
    """True iff ``cfg`` builds exactly the pre-v2 network."""
    c = normalize_structure(cfg)
    return (
        c["trunk"]["activation"] == "tanh"
        and c["trunk"]["skip"] == "none"
        and c["trunk"]["norm"] == "none"
        and c["heads"]["mode"] == "shared"
        and c["wall"]["type"] == "power2"
        and not c["wall"].get("params")
        and c["mask"]["active_modes"] == list(range(1, c["m"] + 1))
        and not c["freeze"]["trunk"]
        and not c["freeze"]["heads"]
    )


# ----------------------------------------------------------------------------
def _trunk_params(width: int, depth: int, norm: str = "none") -> int:
    n = width * 1 + width                     # 1 -> width
    n += (depth - 1) * (width * width + width)  # width -> width
    if norm == "layernorm":
        n += (depth - 1) * 2 * width           # weight + bias per block
    return n


def head_rank(cfg: dict) -> int:
    """Low-rank width used by ``tied_grouped`` heads."""
    c = normalize_structure(cfg)
    r = c["heads"]["rank"]
    if r and r > 0:
        return int(r)
    g = max(1, c["heads"]["group_size"])
    return max(1, min(2 * g, 8))


def param_count(cfg: dict) -> int:
    c = normalize_structure(cfg)
    m = c["m"]
    t, h = c["trunk"], c["heads"]
    n = _trunk_params(t["width"], t["depth"], t["norm"])
    groups = h.get("groups") or [list(range(m))]
    mode = h["mode"]
    if mode == "shared":
        n += t["width"] * 2 * m + 2 * m
    elif mode == "tied_grouped":
        r = head_rank(c)
        n += r * t["width"]                          # shared basis (no bias)
        for g in groups:
            n += (2 * len(g)) * r + (2 * len(g))
    else:  # grouped / per_mode
        for g in groups:
            n += t["width"] * 2 * len(g) + 2 * len(g)
    return int(n)


def describe(cfg: dict) -> str:
    c = normalize_structure(cfg)
    t, h = c["trunk"], c["heads"]
    parts = [t["activation"], f"w{t['width']}", f"d{t['depth']}", h["mode"],
             c["wall"]["type"], f"m{c['m']}"]
    if t["skip"] != "none":
        parts.append(t["skip"])
    if h["mode"] in ("grouped", "tied_grouped"):
        parts.append("g" + "-".join(str(len(g)) for g in h["groups"]))
    if h.get("tie_weights"):
        parts.append("tie")
    act = c["mask"]["active_modes"]
    if len(act) != c["m"]:
        parts.append("act" + "".join(str(i) for i in act))
    return "/".join(parts)


def canonical_json(cfg: dict) -> str:
    return json.dumps(normalize_structure(cfg), sort_keys=True,
                      separators=(",", ":"))


def structure_hash(cfg: dict) -> str:
    return hashlib.sha256(canonical_json(cfg).encode("utf-8")).hexdigest()[:12]


# ----------------------------------------------------------------------------
MUTATIONS = ("width_up", "width_down", "depth_up", "depth_down", "activation",
             "heads_mode", "group_size", "wall", "skip", "mode_off", "mode_on",
             "rank_up", "rank_down")


def mutate(cfg: dict, op: str, rng: random.Random) -> dict:
    """Apply one discrete structural mutation and return a valid structure."""
    c = normalize_structure(cfg)
    m = c["m"]

    if op == "width_up":
        c["trunk"]["width"] = min(512, int(round(c["trunk"]["width"] * 1.5)))
    elif op == "width_down":
        c["trunk"]["width"] = max(8, int(round(c["trunk"]["width"] * 0.7)))
    elif op == "depth_up":
        c["trunk"]["depth"] = min(8, c["trunk"]["depth"] + 1)
    elif op == "depth_down":
        c["trunk"]["depth"] = max(1, c["trunk"]["depth"] - 1)
    elif op == "activation":
        c["trunk"]["activation"] = rng.choice(ACTIVATIONS)
    elif op == "heads_mode":
        c["heads"]["mode"] = rng.choice(HEAD_MODES)
        c["heads"]["groups"] = None
    elif op == "group_size":
        divs = [d for d in range(1, m + 1) if m % d == 0]
        c["heads"]["group_size"] = rng.choice(divs)
        c["heads"]["groups"] = None
        if c["heads"]["mode"] == "shared":
            c["heads"]["mode"] = "grouped"
    elif op == "wall":
        c["wall"]["type"] = rng.choice(WALL_TYPES)
    elif op == "skip":
        c["trunk"]["skip"] = rng.choice(SKIPS)
    elif op == "mode_off":
        act = [i for i in c["mask"]["active_modes"] if i != rng.choice(
            c["mask"]["active_modes"])]
        if act:
            c["mask"]["active_modes"] = act
    elif op == "mode_on":
        off = [i for i in range(1, m + 1) if i not in c["mask"]["active_modes"]]
        if off:
            c["mask"]["active_modes"] = sorted(
                c["mask"]["active_modes"] + [rng.choice(off)])
    elif op == "rank_up":
        c["heads"]["rank"] = min(max(2, head_rank(c)) + 2, 64)
        c["heads"]["mode"] = "tied_grouped"
    elif op == "rank_down":
        c["heads"]["rank"] = max(1, head_rank(c) - 2)
        c["heads"]["mode"] = "tied_grouped"
    else:
        raise ValueError(f"unknown mutation {op!r}")

    return normalize_structure(c)


def crossover(a: dict, b: dict, rng: random.Random) -> dict:
    """Mode-preserving crossover of two structures (physically sane child)."""
    ca, cb = normalize_structure(a), normalize_structure(b)
    child = json.loads(json.dumps(ca))
    # trunk: take one parent's hidden shape wholesale (width/depth jointly)
    src = cb if rng.random() < 0.5 else ca
    child["trunk"]["width"] = src["trunk"]["width"]
    child["trunk"]["depth"] = src["trunk"]["depth"]
    child["trunk"]["activation"] = src["trunk"]["activation"]
    # wall factor must stay compatible with the case; keep parent A unless B
    if rng.random() < 0.5:
        child["wall"] = json.loads(json.dumps(cb["wall"]))
    # heads: take a valid grouping from one parent if m matches
    if cb["m"] == ca["m"] and rng.random() < 0.5:
        child["heads"] = json.loads(json.dumps(cb["heads"]))
    if rng.random() < 0.5:
        child["mask"] = json.loads(json.dumps(cb["mask"]))
    return normalize_structure(child)


def random_structure(base: dict, rng: random.Random,
                     n_ops: int = 2) -> dict:
    """Apply ``n_ops`` distinct random mutations to ``base``."""
    c = normalize_structure(base)
    ops = rng.sample(MUTATIONS, k=min(n_ops, len(MUTATIONS)))
    for op in ops:
        c = mutate(c, op, rng)
        if validate_structure(c):
            c = normalize_structure(base)
    return c


def sample_valid(base: dict, rng: random.Random, n_ops: int = 2,
                 attempts: int = 20) -> dict:
    """Mutation that is guaranteed valid (falls back to ``base``)."""
    for _ in range(attempts):
        c = random_structure(base, rng, n_ops=n_ops)
        if not validate_structure(c):
            return c
    return normalize_structure(base)


def iter_modes(cfg: dict) -> Iterable[int]:
    return range(cfg["m"])
