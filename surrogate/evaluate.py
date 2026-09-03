#!/usr/bin/env python3
"""Evaluate the stress-field surrogate (plate plan M12).

Three levels of evidence, deliberately independent of the training objective:

1. FIELD ACCURACY -- masked nRMSE per component and for von Mises, as a
   percentage of sigma_gross. This is what the loss optimised, so on its own it
   proves the fit converged and nothing more.

2. THE ENGINEERING QUANTITY -- Kt. A 2% field error means nothing if the peak
   is 20% low, and the peak occupies ~0.04% of the pixels, so a plain field
   loss has almost no incentive to get it right. Kt was kept OUT of the loss so
   this reading stays evidence rather than a restatement of the objective.
   Two versions are reported and they answer different questions:
     Kt vs the grid label -> can the network reproduce what it was shown?
     Kt vs kt_fe          -> how far is the answer from the true FEA peak?
   The gap between them is the export's representation floor (median 2.8%),
   which no grid-based surrogate can beat and which is not the model's fault.

3. PHYSICS -- the equilibrium residual div(sigma) = 0, evaluated on the
   predicted field by finite differences in the interior of the material. The
   network was never trained on this. It asks whether the prediction is a
   physically admissible stress state or merely a picture that looks right.

Failure analysis (question 4) needs the per-case table this writes: every row
carries case_id and the geometry, so any bad prediction traces back to its
(r, xc, yc).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from surrogate.dataset import PlateFieldDataset, read_manifests   # noqa: E402
from surrogate.model import UNet                                   # noqa: E402
from surrogate.train import pick_device                            # noqa: E402

HERE = Path(__file__).resolve().parent


def von_mises(f):
    """Plane-stress von Mises from stacked (sxx, syy, sxy)."""
    sxx, syy, sxy = f[0], f[1], f[2]
    return np.sqrt(sxx ** 2 - sxx * syy + syy ** 2 + 3.0 * sxy ** 2)


def equilibrium_residual(f, mask, dx):
    """|div(sigma)| in the material interior, by central differences.

    Static equilibrium with no body force requires
        d(sxx)/dx + d(sxy)/dy = 0 ,  d(sxy)/dx + d(syy)/dy = 0 .
    Only pixels whose four neighbours are all material are scored, so the
    stencil never straddles the hole boundary and reads a stored zero as if it
    were stress. Returned relative to |grad| so it is a dimensionless measure
    of how badly the two terms fail to cancel.
    """
    sxx, syy, sxy = f[0], f[1], f[2]
    m = mask[0] > 0.5
    interior = m.copy()
    interior[1:-1, 1:-1] &= m[:-2, 1:-1] & m[2:, 1:-1] & m[1:-1, :-2] & m[1:-1, 2:]
    interior[0, :] = interior[-1, :] = interior[:, 0] = interior[:, -1] = False
    d = lambda a, ax: (np.roll(a, -1, axis=ax) - np.roll(a, 1, axis=ax)) / (2 * dx)
    rx = d(sxx, 1) + d(sxy, 0)          # axis 1 = x, axis 0 = y
    ry = d(sxy, 1) + d(syy, 0)
    res = np.sqrt(rx ** 2 + ry ** 2)
    scale = np.sqrt(d(sxx, 1) ** 2 + d(syy, 0) ** 2 + d(sxy, 1) ** 2 + d(sxy, 0) ** 2)
    denom = np.median(scale[interior]) if interior.any() else 1.0
    return float(np.mean(res[interior]) / max(denom, 1e-9)), int(interior.sum())


def evaluate_split(net, split, device, rows=None) -> list:
    ds = PlateFieldDataset(split, rows=rows)
    out = []
    net.eval()
    for i in range(len(ds)):
        it = ds[i]
        x = torch.from_numpy(it["x"][None]).to(device)
        with torch.no_grad():
            pred = net(x)[0].cpu().numpy()
        m = it["mask"]
        pred = pred * m                              # nothing lives outside the material
        true = it["y"]
        mb = np.repeat(m > 0.5, 3, axis=0)
        row = {"case_id": it["case_id"], "split": split if isinstance(split, str) else "",
               "r": it["r"], "xc": it["xc"], "yc": it["yc"], "a": it["a"], "b": it["b"],
               "min_ligament": min(it["xc"] - it["r"], it["a"] - it["xc"] - it["r"],
                                   it["yc"] - it["r"], it["b"] - it["yc"] - it["r"])}
        for k, c in enumerate(("sxx", "syy", "sxy")):
            sel = m[0] > 0.5
            row[f"nrmse_{c}"] = float(np.sqrt(np.mean((pred[k][sel] - true[k][sel]) ** 2)) * 100)
        row["nrmse_all"] = float(np.sqrt(np.mean((pred - true)[mb] ** 2)) * 100)
        vm_p, vm_t = von_mises(pred) * (m[0] > 0.5), von_mises(true) * (m[0] > 0.5)
        row["nrmse_vm"] = float(np.sqrt(np.mean((vm_p - vm_t)[m[0] > 0.5] ** 2)) * 100)
        # Kt: peak / sigma_net, with the target's own peak and the true FEA peak
        s_ratio = it["sigma_gross"] / it["sigma_net"]
        row["kt_pred"] = float(vm_p.max() * s_ratio)
        row["kt_label"] = float(vm_t.max() * s_ratio)
        row["kt_fe"] = it["kt_fe"]
        row["kt_err_vs_label"] = 100 * (row["kt_pred"] - row["kt_label"]) / row["kt_label"]
        row["kt_err_vs_fe"] = 100 * (row["kt_pred"] - row["kt_fe"]) / row["kt_fe"]
        dx = it["a"] / 384.0 if it["a"] == 200.0 else 200.0 / 384.0
        row["equil_pred"], row["n_interior"] = equilibrium_residual(pred, m, dx / it["b"])
        row["equil_true"], _ = equilibrium_residual(true, m, dx / it["b"])
        out.append(row)
    return out


def summarise(rows, name) -> dict:
    g = lambda k: np.array([r[k] for r in rows])
    return {"split": name, "n": len(rows),
            "nrmse_all": float(np.mean(g("nrmse_all"))),
            "nrmse_vm": float(np.mean(g("nrmse_vm"))),
            "kt_mae_label": float(np.mean(np.abs(g("kt_err_vs_label")))),
            "kt_bias_label": float(np.mean(g("kt_err_vs_label"))),
            "kt_mae_fe": float(np.mean(np.abs(g("kt_err_vs_fe")))),
            "kt_bias_fe": float(np.mean(g("kt_err_vs_fe"))),
            "kt_p95_fe": float(np.percentile(np.abs(g("kt_err_vs_fe")), 95)),
            "equil_pred": float(np.mean(g("equil_pred"))),
            "equil_true": float(np.mean(g("equil_true")))}


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate the surrogate (plate plan M12).")
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--splits", nargs="+",
                    default=["train", "val", "test", "ood_large_r", "ood_aspect"])
    args = ap.parse_args()

    ckpt_path = HERE / "runs" / args.tag / "best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    dev = pick_device(args.device)
    net = UNet(width=ckpt["width"]).to(dev)
    net.load_state_dict(ckpt["model"])
    print(f"checkpoint {ckpt_path}  (epoch {ckpt['epoch']}, val loss {ckpt['val_loss']:.5f})")

    rows_all, summaries = [], []
    manifest_rows = read_manifests()
    for sp in args.splits:
        rows = evaluate_split(net, sp, dev, rows=manifest_rows)
        for r in rows:
            r["split"] = sp
        rows_all += rows
        summaries.append(summarise(rows, sp))

    print(f"\n{'split':<14}{'n':>4} {'nRMSE all':>10} {'nRMSE vM':>9} "
          f"{'Kt MAE(lbl)':>12} {'Kt MAE(FE)':>11} {'Kt bias(FE)':>12} {'equil':>8}")
    for s in summaries:
        print(f"{s['split']:<14}{s['n']:>4} {s['nrmse_all']:9.3f}% {s['nrmse_vm']:8.3f}% "
              f"{s['kt_mae_label']:11.2f}% {s['kt_mae_fe']:10.2f}% {s['kt_bias_fe']:+11.2f}% "
              f"{s['equil_pred']:8.4f}")
    print(f"\n(equilibrium residual of the FEA labels themselves, for reference: "
          f"{np.mean([s['equil_true'] for s in summaries]):.4f})")

    out = HERE / "runs" / args.tag
    with (out / "per_case.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_all[0]))
        w.writeheader(); w.writerows(rows_all)
    with (out / "summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summaries[0]))
        w.writeheader(); w.writerows(summaries)
    print(f"\nper-case table: {out / 'per_case.csv'}   (every row carries case_id + geometry)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
