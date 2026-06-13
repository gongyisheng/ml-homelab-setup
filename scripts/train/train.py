"""Self-contained reference training entry: a tiny GPT trained on synthetic tokens.

Torch-only (no downloads), so the launchers run end-to-end on any box. Demonstrates
the DDP path (gradient all-reduce) and rank-0 checkpointing. Swap in your real training
script via TRAIN_SCRIPT in the launchers.

Run directly (single GPU) or under torchrun (multi-GPU / multi-node) — it reads the
RANK / LOCAL_RANK / WORLD_SIZE env vars torchrun sets.
"""
import argparse
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


class TinyGPT(nn.Module):
    def __init__(self, vocab, d_model, n_layer, n_head, seq_len):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(seq_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_head, d_model * 4, batch_first=True)
        self.blocks = nn.TransformerEncoder(layer, n_layer)
        self.head = nn.Linear(d_model, vocab)
        self.seq_len = seq_len

    def forward(self, x):
        pos = torch.arange(x.size(1), device=x.device)
        h = self.tok(x) + self.pos(pos)
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1), x.device)
        return self.head(self.blocks(h, mask=mask, is_causal=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layer", type=int, default=4)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--vocab", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ckpt-dir", default="/tmp/homelab-train")
    args = ap.parse_args()

    ddp = "RANK" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    if ddp:
        dist.init_process_group(backend="nccl")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(local_rank) if torch.cuda.is_available() else None

    model = TinyGPT(args.vocab, args.d_model, args.n_layer, args.n_head, args.seq_len).to(device)
    if ddp:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    gen = torch.Generator(device="cpu").manual_seed(rank)
    for step in range(args.steps):
        batch = torch.randint(0, args.vocab, (args.batch_size, args.seq_len + 1), generator=gen)
        x, y = batch[:, :-1].to(device), batch[:, 1:].to(device)
        logits = model(x)
        loss = loss_fn(logits.reshape(-1, args.vocab), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if rank == 0 and (step % 10 == 0 or step == args.steps - 1):
            print(f"step {step:4d} | loss {loss.item():.4f}")

    if rank == 0:
        os.makedirs(args.ckpt_dir, exist_ok=True)
        sd = model.module.state_dict() if ddp else model.state_dict()
        path = os.path.join(args.ckpt_dir, "ckpt.pt")
        torch.save(sd, path)
        print(f"saved checkpoint -> {path}")

    if ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
