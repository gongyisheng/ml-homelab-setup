import os

import torch
import torch.distributed as dist


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    # Pass device_id so collectives bind to this rank's GPU (also silences the
    # barrier() device-context warning).
    dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))

    # Each rank starts with rank+1; the all-reduce sum should be 1+2+...+world_size.
    local_val = rank + 1.0
    x = torch.tensor([local_val], device=local_rank)
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    total = x.item()

    # Print in rank order so the lines don't interleave.
    for r in range(world_size):
        if rank == r:
            print(f"[rank {rank}/{world_size}] local {local_val:.0f} -> all_reduce sum {total:.0f}", flush=True)
        dist.barrier()

    if rank == 0:
        expected = world_size * (world_size + 1) / 2
        ok = abs(total - expected) < 1e-3
        print(f"\nall_reduce sum = {total:.0f} (expected {expected:.0f}) -> {'PASS' if ok else 'FAIL'}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
