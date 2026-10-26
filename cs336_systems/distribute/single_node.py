import json
import timeit
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import tyro

from cs336_systems.distribute.utils import DCConfig, get_tensor_size, setup, sync


def communicate(rank: int, cfg: DCConfig, exp_dir: Path):
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

    # save results
    if rank == 0:
        avg_time = sum(timings) / len(timings)
        print(f"[{cfg.data_size}] Avg time ({cfg.world_size} ranks, {cfg.backend}): {avg_time:.6f}s")

        results = {
            "avg_time_sec": avg_time,
            "min_time_sec": min(timings),
            "max_time_sec": max(timings),
            "timings_by_rank": timings,
        }

        # save results
        (exp_dir / "results.json").write_text(json.dumps(results, indent=4))
        print(f"Artifacts saved to: {exp_dir.resolve()}")

    dist.destroy_process_group()


def main(cfg: DCConfig) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = cfg.save_base_dir / f"ws{cfg.world_size}_{cfg.backend}_{cfg.data_size}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    (exp_dir / "args.json").write_text(cfg.model_dump_json(indent=4))

    mp.spawn(fn=communicate, args=(cfg, exp_dir), nprocs=cfg.world_size, join=True)


if __name__ == "__main__":
    cfg = tyro.cli(DCConfig)
    main(cfg)
