import os
import torch
import torch.distributed as dist


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)

    # Each rank starts with rank+1; the all-reduce sum should be world_size*(world_size+1)/2.
    x = torch.tensor([rank + 1.0], device=local_rank)
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    expected = world_size * (world_size + 1) / 2
    ok = abs(x.item() - expected) < 1e-3
    print(f"[Rank {rank}] after all_reduce: {x.item()} (expected {expected}) {'OK' if ok else 'MISMATCH'}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
