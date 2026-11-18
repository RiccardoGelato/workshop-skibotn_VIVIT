"""Fine-tune ViViT (LoRA, optional nofe tubelets) on your own clips.

    python training.py --mode hybrid --train-csv train.csv --val-csv val.csv --num-labels 5

CSV columns: path,label   (label = integer class id, with a header row)
Modes: baseline | hybrid | full   (see model.py)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import av
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model import MODES, build_model, extra_state, is_nofe_param


# --------------------------------------------------------------------------- preprocessing


def read_frames(path: str, num_frames: int = 32, stride: int = 4) -> list[np.ndarray]:
    """Centered clip of ``num_frames`` RGB frames, every ``stride``-th frame.

    Decodes the whole file into memory, which is fine for short clips.
    """
    container = av.open(path)
    frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    total = len(frames)
    span = num_frames * stride
    start = max(0, (total - span) // 2)
    idx = np.clip(np.arange(start, start + span, stride), 0, total - 1)
    return [frames[i] for i in idx]


def preprocess(frames: list[np.ndarray], size: int = 224, resize: int = 256) -> torch.Tensor:
    """(T, 3, size, size) float tensor in [-1, 1]. Put your own preprocessing here."""
    x = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 255.0  # (T, 3, H, W)
    x = F.interpolate(x, size=(resize, resize), mode="bilinear", align_corners=False)
    top = left = (resize - size) // 2
    x = x[:, :, top : top + size, left : left + size]
    return (x - 0.5) / 0.5


class ClipDataset(Dataset):
    def __init__(self, csv_path: str):
        with open(csv_path, newline="") as fh:
            rows = list(csv.DictReader(fh))
        self.paths = [r["path"] for r in rows]
        self.labels = [int(r["label"]) for r in rows]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        return preprocess(read_frames(self.paths[i])), self.labels[i]


# --------------------------------------------------------------------------- training


def make_optimizer(model, lr: float, lr_nofe: float, weight_decay: float):
    """LoRA and classifier at ``lr``; the new tubelet module (trained from scratch) at ``lr_nofe``."""
    nofe, rest = [], []
    for name, param in model.named_parameters():
        if param.requires_grad:
            (nofe if is_nofe_param(name) else rest).append(param)
    groups = [{"params": rest, "lr": lr}]
    if nofe:
        groups.append({"params": nofe, "lr": lr_nofe})
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        logits = model(pixel_values=x.to(device)).logits
        correct += (logits.argmax(-1).cpu() == y).sum().item()
        total += len(y)
    return correct / max(total, 1)


def train(model, train_loader, val_loader, device, *, epochs, lr, lr_nofe, weight_decay, amp):
    optimizer = make_optimizer(model, lr, lr_nofe, weight_decay)
    use_amp = amp and device.type == "cuda"
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    best = -1.0
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                loss = model(pixel_values=x, labels=y).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            running += loss.item()
        acc = evaluate(model, val_loader, device) if val_loader else float("nan")
        print(f"epoch {epoch:3d}  loss {running / len(train_loader):.4f}  val_acc {acc:.4f}")
        if val_loader and acc > best:
            best = acc
            params = dict(model.named_parameters())
            best_state = {n: params[n].detach().cpu().clone() for n in trainable}
    if best_state is not None:
        model.load_state_dict(best_state, strict=False)  # restore the best epoch's trainable weights
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=MODES, default="hybrid")
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--val-csv")
    ap.add_argument("--num-labels", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4, help="LoRA and classifier")
    ap.add_argument("--lr-nofe", type=float, default=1e-3, help="new nofe tubelet module")
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--width", type=int, default=8, help="KernelNN width; memory grows with width^2")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--amp", action="store_true", help="bfloat16 autocast on CUDA")
    ap.add_argument("--out", default="runs/vivit_lora")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.num_labels, args.mode, lora_r=args.lora_r, width=args.width).to(device)
    model.print_trainable_parameters()

    train_loader = DataLoader(ClipDataset(args.train_csv), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers)
    val_loader = (DataLoader(ClipDataset(args.val_csv), batch_size=args.batch_size,
                             num_workers=args.workers) if args.val_csv else None)

    train(model, train_loader, val_loader, device, epochs=args.epochs, lr=args.lr,
          lr_nofe=args.lr_nofe, weight_decay=args.weight_decay, amp=args.amp)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)  # LoRA adapter + classifier
    torch.save({"mode": args.mode, "state": extra_state(model)}, out / "extra.pt")  # nofe tubelets
    print(f"saved to {out}")


if __name__ == "__main__":
    main()