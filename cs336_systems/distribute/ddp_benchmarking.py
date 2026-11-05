import json
from datetime import datetime
from pathlib import Path
from typing import Annotated

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import tyro
from cs336_basics.layers import TransformerLM, cross_entropy
from cs336_basics.train_loop import AdamW
from pydantic import BaseModel, Field
from torch.profiler import ProfilerActivity, profile, record_function

from cs336_systems.distribute import FSDP, BaseDDP, OptShard, get_ddp_cls
from cs336_systems.distribute.utils import DCConfig, generate_data, memory_profile_json, setup, summarize_metrics, sync, time_profile_json
from cs336_systems.utils import ModelArgs, ModelConfig


class DDPBenchmarkArgs(BaseModel):
    dc: Annotated[DCConfig, tyro.conf.OmitArgPrefixes] = Field(default_factory=DCConfig)
    model: Annotated[ModelArgs, tyro.conf.OmitArgPrefixes] = Field(default_factory=ModelArgs)


def run_benchmark(rank: int, dc_cfg: DCConfig, model_cfg: ModelConfig, exp_dir: Path, ddp_cls: type[BaseDDP]):
    setup(rank, dc_cfg.world_size, dc_cfg.master_addr, dc_cfg.master_port, dc_cfg.backend)

    curr_device = torch.device(f"cuda:{rank}" if dc_cfg.device == "cuda" else "cpu")
    if dc_cfg.device == "cuda":
        torch.cuda.set_device(curr_device)

    # init model and optimizer
    raw_model = TransformerLM(
        model_cfg.vocab_size,
        model_cfg.context_length,
        model_cfg.num_layers,
        model_cfg.d_model,
        model_cfg.num_heads,
        model_cfg.d_ff,
        model_cfg.rope_theta,
        curr_device,
        model_cfg.dtype,
    )

    # ddp wrap
    if dc_cfg.fsdp:
        model = FSDP(raw_model, compute_dtype=model_cfg.dtype)
    else:
        model = ddp_cls(raw_model)

    if dc_cfg.opt_shard:
        optimizer = OptShard(model.parameters(), AdamW, lr=1e-4, weight_decay=0.1)
    else:
        optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)

    # data
    x, targets = generate_data(model_cfg, dc_cfg, curr_device)

    # warm up
    for _ in range(dc_cfg.warm_up):
        optimizer.zero_grad()
        logits = model(x)
        loss = cross_entropy(logits, targets)
        loss.backward()
        model.finish_grad_sync()
        optimizer.step()

    sync(dc_cfg.device, rank)
    dist.barrier()

    # measure steps
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    trace_path = exp_dir / f"trace_rank_{rank}.json"
    activities = [ProfilerActivity.CPU]
    if dc_cfg.device == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(activities=activities, record_shapes=False, profile_memory=dc_cfg.record_memory, acc_events=True) as prof:
        start_event.record()
        for _ in range(dc_cfg.steps):
            optimizer.zero_grad()

            with record_function("FORWARD"):
                logits = model(x)
                loss = cross_entropy(logits, targets)

            with record_function("BACKWARD"):
                loss.backward()

            with record_function("EXPOSED_COMM_WAIT"):
                model.finish_grad_sync()

            with record_function("OPTIMIZER_STEP"):
                optimizer.step()

        end_event.record()

    sync(dc_cfg.device, rank)
    dist.barrier()

    total_step_time_ms = start_event.elapsed_time(end_event) / dc_cfg.steps

    # export and parse trace
    prof.export_chrome_trace(str(trace_path))
    metrics: dict[str, float] = time_profile_json(trace_path)
    if dc_cfg.record_memory:
        metrics.update(memory_profile_json(trace_path))

    metrics.update(
        {
            "total_step_time_ms": total_step_time_ms,
            "actual_comm_per_step_ms": metrics["actual_comm_time_ms"] / dc_cfg.steps,
            "exposed_comm_per_step_ms": metrics["exposed_comm_time_ms"] / dc_cfg.steps,
            "comm_ratio_pct": ((metrics["actual_comm_time_ms"] / dc_cfg.steps) / total_step_time_ms * 100.0),
        }
    )

    # gather
    gathered_metrics: list[dict[str, float]] = [{} for _ in range(dc_cfg.world_size)]
    dist.all_gather_object(gathered_metrics, metrics)

    if rank == 0:
        results_path = exp_dir / "results.json"
        results = {
            "summary": summarize_metrics(gathered_metrics),
            "per_rank_metrics": gathered_metrics,
        }
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)

        print(f"\n[✓] Benchmark finished successfully ({dc_cfg.world_size} ranks).")
        print(f"[✓] Detailed results written to: {results_path.resolve()}\n")

    dist.destroy_process_group()


def main(args: DDPBenchmarkArgs) -> None:
    dc_cfg = args.dc
    model_cfg = args.model.build_model_config()
    ddp_cls = get_ddp_cls(dc_cfg.ddp_type)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = dc_cfg.save_base_dir / f"ws{dc_cfg.world_size}_{dc_cfg.backend}_{dc_cfg.ddp_type}_{args.model.model_size}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    (exp_dir / "args.json").write_text(args.model_dump_json(indent=4))

    mp.spawn(fn=run_benchmark, args=(dc_cfg, model_cfg, exp_dir, ddp_cls), nprocs=dc_cfg.world_size, join=True)


if __name__ == "__main__":
    args = tyro.cli(DDPBenchmarkArgs)
    main(args)
