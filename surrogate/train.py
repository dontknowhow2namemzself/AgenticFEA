#!/usr/bin/env python3
"""Train the stress-field surrogate (plate plan M12).

Deliberate choices, all of them things a reviewer will ask about:

* The loss is the MASKED MSE over the three stress components and nothing else.
  Kt error and the equilibrium residual are NOT in the loss, on purpose: they
  are the two independent checks this project reports, and a metric you have
  optimised is no longer evidence. Whether a plain field loss is enough to get
  the peak right is a question this baseline exists to answer.

* The split comes from the manifest. `train` fits, `val` picks the checkpoint,
  and nothing else is touched until the final evaluation.

* Every case is a separate sample; there is no pixel-level shuffling. Pixels
  within one field are strongly correlated, so splitting by pixel would leak
  the test set into training.

Unit system: dimensionless (stress / sigma_gross).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from surrogate.dataset import PlateFieldDataset, read_manifests   # noqa: E402
from surrogate.model import UNet, count_params, masked_mse_loss   # noqa: E402

HERE = Path(__file__).resolve().parent
CKPT_DIR = HERE / "runs"


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Batcher:
    """Minimal batching over an in-memory split (the whole set is ~0.5 GB)."""

    def __init__(self, ds, batch_size, device, shuffle=True, seed=0):
        self.items = [ds[i] for i in range(len(ds))]     # load once, reuse each epoch
        self.bs, self.device, self.shuffle = batch_size, device, shuffle
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return (len(self.items) + self.bs - 1) // self.bs

    def __iter__(self):
        order = (self.rng.permutation(len(self.items)) if self.shuffle
                 else np.arange(len(self.items)))
        for k in range(0, len(order), self.bs):
            chunk = [self.items[i] for i in order[k:k + self.bs]]
            yield (torch.from_numpy(np.stack([c["x"] for c in chunk])).to(self.device),
                   torch.from_numpy(np.stack([c["y"] for c in chunk])).to(self.device),
                   torch.from_numpy(np.stack([c["mask"] for c in chunk])).to(self.device),
                   [c["case_id"] for c in chunk])


def evaluate_loss(net, batcher) -> float:
    net.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for x, y, m, _ in batcher:
            loss = masked_mse_loss(net(x), y, m)
            tot += loss.item() * x.shape[0]
            n += x.shape[0]
    return tot / max(n, 1)


def train(epochs=60, batch_size=8, lr=2e-3, width=32, seed=0, device="auto",
          out_dir=CKPT_DIR, tag="baseline") -> dict:
    torch.manual_seed(seed)
    dev = pick_device(device)
    rows = read_manifests()
    tr = Batcher(PlateFieldDataset("train", rows=rows), batch_size, dev, seed=seed)
    va = Batcher(PlateFieldDataset("val", rows=rows), batch_size, dev, shuffle=False)
    net = UNet(width=width).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(tr))

    out_dir = Path(out_dir) / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device {dev} | params {count_params(net):,} | "
          f"train {len(tr.items)} cases, val {len(va.items)} | {epochs} epochs")
    best, history, t0 = float("inf"), [], time.time()
    for ep in range(1, epochs + 1):
        net.train()
        run, n = 0.0, 0
        for x, y, m, _ in tr:
            opt.zero_grad(set_to_none=True)
            loss = masked_mse_loss(net(x), y, m)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            run += loss.detach().item() * x.shape[0]
            n += x.shape[0]
        tr_loss = run / n
        va_loss = evaluate_loss(net, va)
        history.append({"epoch": ep, "train": tr_loss, "val": va_loss,
                        "lr": sched.get_last_lr()[0]})
        star = ""
        if va_loss < best:
            best = va_loss
            torch.save({"model": net.state_dict(), "width": width, "epoch": ep,
                        "val_loss": va_loss}, out_dir / "best.pt")
            star = "  <- best"
        if ep <= 3 or ep % 5 == 0 or ep == epochs:
            print(f"  epoch {ep:3d}/{epochs}  train {tr_loss:.5f}  val {va_loss:.5f}"
                  f"  rmse {np.sqrt(va_loss):.4f}{star}")
    dt = time.time() - t0
    (out_dir / "history.json").write_text(json.dumps(
        {"history": history, "best_val": best, "seconds": dt,
         "config": {"epochs": epochs, "batch_size": batch_size, "lr": lr,
                    "width": width, "seed": seed, "loss": "masked_mse"}}, indent=2))
    print(f"done in {dt/60:.1f} min | best val loss {best:.5f} "
          f"(masked RMSE {np.sqrt(best):.4f} of sigma_gross)")
    return {"best_val": best, "out_dir": str(out_dir), "seconds": dt}


def main() -> int:
    ap = argparse.ArgumentParser(description="Train the field surrogate (plate plan M12).")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tag", default="baseline")
    args = ap.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, width=args.width,
          seed=args.seed, device=args.device, tag=args.tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
