#!/usr/bin/env python3
"""Dataset for the stress-field surrogate (plate plan M12).

Reads the .npz tensors written by export_field.py and hands the network:

    input   (2, H, W)   mask, sdf/b          -- what the geometry is
    target  (3, H, W)   sxx, syy, sxy        -- already divided by sigma_gross
    mask    (1, H, W)   which pixels count

Three conventions are enforced HERE, in one place, so training and evaluation
can never drift apart:

1. THE SPLIT IS READ, NEVER COMPUTED. `split` was written into the manifest at
   generation time. A held-out set that has been used to tune anything is not
   held out any more, so this loader offers no way to re-shuffle it.

2. EVERY AVERAGE IS MASK-WEIGHTED. The canvas is a fixed physical window, so
   the fraction of it covered by material varies per case (91% for a small
   centred hole, 61% for the shorter out-of-distribution plate). Averaging over
   all pixels would divide by a different denominator for every case and make
   the same model look better wherever more of the canvas is empty --
   measured: an identical 0.10 error reads as 0.095 in-distribution and 0.078
   out-of-distribution. The mask is carried alongside the target so the loss
   and the metrics use the same denominator.

3. THE SDF IS SCALED BY THE PLATE WIDTH b. Raw values run to +166 mm; feeding
   that into a convolution stack is asking for large activations and slow
   convergence. Dividing by b keeps it O(1) and gives it a readable meaning
   ("how many plate widths from the hole"). The targets need no scaling -- the
   export already divided them by sigma_gross.

Unit system: the arrays are dimensionless (stress / sigma_gross, length / b).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DATA_ROOTS = (PROJECT / "out" / "dataset", PROJECT / "out" / "dataset_ood_aspect")

CHANNELS_IN = ("mask", "sdf")
CHANNELS_OUT = ("sxx", "syy", "sxy")


def read_manifests(roots=DATA_ROOTS) -> list:
    """Every generated case, with its split and the path to its tensors."""
    rows = []
    for root in roots:
        mpath = Path(root) / "manifest.csv"
        if not mpath.exists():
            continue
        for r in csv.DictReader(mpath.open()):
            if r.get("status") != "ok" or not r.get("npz"):
                continue                      # a case that failed a gate is not data
            r["root"] = str(root)
            r["npz_path"] = str(Path(root) / r["case_id"] / r["npz"])
            rows.append(r)
    if not rows:
        raise FileNotFoundError(f"no exported cases under {[str(r) for r in roots]}; "
                                f"run batch_run.py --export first")
    return rows


def load_case(npz_path) -> dict:
    """One case as float32 arrays, already scaled the way the network wants."""
    d = np.load(npz_path)
    b = float(d["b"])
    mask = d["mask"].astype(np.float32)
    x = np.stack([mask, d["sdf"].astype(np.float32) / b], axis=0)
    y = np.stack([d[c].astype(np.float32) for c in CHANNELS_OUT], axis=0)
    y = y * mask                              # nothing outside the material is a target
    return {"x": x, "y": y, "mask": mask[None, :, :],
            "kt_fe": float(d["kt_fe"]), "sigma_gross": float(d["sigma_gross"]),
            "sigma_net": float(d["sigma_net"]), "r": float(d["r"]),
            "xc": float(d["xc"]), "yc": float(d["yc"]), "a": float(d["a"]), "b": b}


class PlateFieldDataset:
    """Cases of one split. Torch-free so it can be self-tested without a GPU."""

    def __init__(self, split, roots=DATA_ROOTS, rows=None):
        splits = {split} if isinstance(split, str) else set(split)
        self.rows = [r for r in (rows or read_manifests(roots)) if r["split"] in splits]
        if not self.rows:
            raise ValueError(f"no cases for split(s) {sorted(splits)}")
        self.split = splits

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        item = load_case(row["npz_path"])
        item["case_id"] = row["case_id"]      # kept end to end: a bad prediction
        item["root"] = row["root"]            # must be traceable back to its geometry
        return item


def masked_mse(pred, true, mask) -> float:
    """Mean squared error over material pixels only (numpy; the torch loss mirrors it)."""
    m = mask.astype(bool)
    if pred.ndim == 3:                        # (C, H, W) -> broadcast the mask
        m = np.repeat(m, pred.shape[0], axis=0) if m.shape[0] == 1 else m
    return float(((pred - true)[m] ** 2).mean())


def _selftest() -> bool:
    print("=" * 68)
    print("surrogate/dataset.py acceptance")
    print("=" * 68)
    rows = read_manifests()
    counts = {}
    for r in rows:
        counts[r["split"]] = counts.get(r["split"], 0) + 1
    print(f"cases with exported tensors: {len(rows)}")
    for k in sorted(counts):
        print(f"   {k:<14} {counts[k]:>4}")

    ds = PlateFieldDataset("train")
    item = ds[0]
    print(f"\nsample from train: {item['case_id']}  r={item['r']:.2f} "
          f"centre ({item['xc']:.1f}, {item['yc']:.1f})")
    print(f"   x {item['x'].shape} {item['x'].dtype}   y {item['y'].shape}   mask {item['mask'].shape}")
    print(f"   input  ch0 (mask) range [{item['x'][0].min():.2f}, {item['x'][0].max():.2f}]")
    print(f"   input  ch1 (sdf/b) range [{item['x'][1].min():.3f}, {item['x'][1].max():.3f}]")
    for k, c in enumerate(CHANNELS_OUT):
        print(f"   target {c} range [{item['y'][k].min():.3f}, {item['y'][k].max():.3f}]")
    mfrac = item["mask"].mean()
    print(f"   material fraction {mfrac * 100:.1f}%")

    # --- acceptance gates ---
    splits_seen = set(counts)
    all_ids = [r["case_id"] + "@" + r["root"] for r in rows]
    train_ids = {r["case_id"] + "@" + r["root"] for r in rows if r["split"] == "train"}
    held_ids = {r["case_id"] + "@" + r["root"] for r in rows
                if r["split"] in ("test", "ood_large_r", "ood_aspect")}
    sdf_ok = all(abs(PlateFieldDataset("val")[i]["x"][1]).max() < 3.0 for i in range(3))
    zero_outside = True
    for i in range(3):
        it = ds[i]
        outside = it["mask"][0] < 0.5
        zero_outside &= bool(np.all(it["y"][:, outside] == 0))

    checks = [
        ("all five splits present",
         splits_seen == {"train", "val", "test", "ood_large_r", "ood_aspect"}),
        ("no case appears twice", len(all_ids) == len(set(all_ids))),
        ("train and held-out sets are disjoint", not (train_ids & held_ids)),
        ("failed cases excluded (every row has tensors)",
         all(Path(r["npz_path"]).exists() for r in rows)),
        ("sdf scaled to O(1)", sdf_ok),
        ("targets are exactly zero outside the material", zero_outside),
        ("ood_aspect really is a different plate",
         {float(np.load(r["npz_path"])["a"]) for r in rows if r["split"] == "ood_aspect"}
         != {float(np.load(r["npz_path"])["a"]) for r in rows if r["split"] == "train"}),
    ]
    print("-" * 68)
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    passed = all(ok for _, ok in checks)
    print("-" * 68)
    print(f"ACCEPTANCE: {'PASS' if passed else 'FAIL'}")
    print("=" * 68)
    return passed


if __name__ == "__main__":
    sys.exit(0 if _selftest() else 1)
