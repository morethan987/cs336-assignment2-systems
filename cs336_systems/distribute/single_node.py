import timeit

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from utils import DCConfig, get_tensor_size, parse_args, setup, sync


def communicate(rank: int, cfg: DCConfig):
    setup(rank, cfg.world_size, cfg.master_addr, cfg.master_port, cfg.backend)

    curr_device = torch.device(f"cuda:{rank}" if cfg.device == "cuda" else "cpu")
    data = torch.randint(0, 10, (get_tensor_size(cfg.data_size),), device=curr_device)

    # warm up
    for _ in range(cfg.warm_up):
        dist.all_reduce(data, async_op=False)

    # timing
    sync(cfg.device, rank)
    dist.barrier()
    start = timeit.default_timer()

    for _ in range(cfg.steps):
        dist.all_reduce(data, async_op=False)

    sync(cfg.device, rank)
    local_t = (timeit.default_timer() - start) / cfg.steps

    # collecting
    timings: list[float] = [0.0] * cfg.world_size
    dist.all_gather_object(timings, local_t)

    if rank == 0:
        avg_time = sum(timings) / len(timings)
        print(f"[{cfg.data_size}] Avg time ({cfg.world_size} ranks, {cfg.backend}): {avg_time:.6f}s")

    dist.destroy_process_group()


def main(cfg: DCConfig) -> None:
    mp.spawn(fn=communicate, args=(cfg,), nprocs=cfg.world_size, join=True)


if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)
