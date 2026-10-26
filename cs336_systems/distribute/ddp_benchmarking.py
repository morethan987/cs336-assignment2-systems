import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from cs336_basics.layers import TransformerLM, cross_entropy
from cs336_basics.train_loop import AdamW
from torch.profiler import ProfilerActivity, profile, record_function

from cs336_systems.distribute.ddp import NaiveDDP
from cs336_systems.distribute.utils import DCConfig, create_dc_config, generate_data, setup, setup_dc_benchmark_args, sync
from cs336_systems.utils import ModelConfig, create_model_config, setup_model_args


def merge_intervals(
    intervals: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for cur_s, cur_e in intervals[1:]:
        last_s, last_e = merged[-1]
        if cur_s <= last_e:
            merged[-1] = (last_s, max(last_e, cur_e))
        else:
            merged.append((cur_s, cur_e))
    return merged


def subtract_intervals(comm: list[tuple[float, float]], comp: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Exposed communication time"""
    exposed = []
    i = 0
    j = 0
    while i < len(comm):
        c_start, c_end = comm[i]
        while j < len(comp) and comp[j][1] <= c_start:
            j += 1

        curr_start = c_start
        k = j
        while k < len(comp) and comp[k][0] < c_end:
            comp_s, comp_e = comp[k]
            if comp_s > curr_start:
                exposed.append((curr_start, min(comp_s, c_end)))
            curr_start = max(curr_start, comp_e)
            if curr_start >= c_end:
                break
            k += 1

        if curr_start < c_end:
            exposed.append((curr_start, c_end))
        i += 1
    return exposed


def analyze_trace_json(trace_path: Path) -> dict[str, float]:
    with open(trace_path, encoding="utf-8") as f:
        data = json.load(f)

    comm_raw = []
    comp_raw = []

    for event in data.get("traceEvents", []):
        if event.get("ph") == "X" and "ts" in event and "dur" in event:
            cat = event.get("cat", "")
            name = event.get("name", "")

            if cat in ["kernel", "gpu_user_annotation"]:
                start = float(event["ts"])
                end = start + float(event["dur"])

                if "ncclKernel" in name or "nccl" in name.lower():
                    comm_raw.append((start, end))
                else:
                    comp_raw.append((start, end))

    merged_comm = merge_intervals(comm_raw)
    merged_comp = merge_intervals(comp_raw)

    actual_comm_us = sum(e - s for s, e in merged_comm)

    exposed_comm = subtract_intervals(merged_comm, merged_comp)
    exposed_comm_us = sum(e - s for s, e in exposed_comm)

    actual_comm_ms = actual_comm_us / 1000.0
    exposed_comm_ms = exposed_comm_us / 1000.0
    overlap_ratio = ((actual_comm_ms - exposed_comm_ms) / actual_comm_ms * 100.0) if actual_comm_ms > 0 else 0.0

    return {
        "actual_comm_time_ms": actual_comm_ms,
        "exposed_comm_time_ms": exposed_comm_ms,
        "overlap_ratio_pct": overlap_ratio,
    }


def run_benchmark(rank: int, dc_cfg: DCConfig, model_cfg: ModelConfig, exp_dir: Path):
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
        model_cfg.device,
        model_cfg.dtype,
    )
    model = NaiveDDP(raw_model)
    optimizer = AdamW(raw_model.parameters(), lr=1e-4, weight_decay=0.1)

    # data
    x, targets = generate_data(model_cfg, dc_cfg)

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
    trace_path = exp_dir / f"trace_rank_{rank}.json"
    activities = [ProfilerActivity.CPU]
    if dc_cfg.device == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        record_shapes=False,
        profile_memory=False,
    ) as prof:
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

    sync(dc_cfg.device, rank)
    dist.barrier()

    # export and parse trace
    prof.export_chrome_trace(str(trace_path))
    metrics = analyze_trace_json(trace_path)

    # average
    metrics["actual_comm_time_per_step_ms"] = metrics["actual_comm_time_ms"] / dc_cfg.steps
    metrics["exposed_comm_time_per_step_ms"] = metrics["exposed_comm_time_ms"] / dc_cfg.steps

    # gather
    gathered_metrics: list[dict[str, float]] = [{} for _ in range(dc_cfg.world_size)]
    dist.all_gather_object(gathered_metrics, metrics)

    if rank == 0:
        avg_comm = sum(m["actual_comm_time_per_step_ms"] for m in gathered_metrics) / len(gathered_metrics)
        avg_exposed = sum(m["exposed_comm_time_per_step_ms"] for m in gathered_metrics) / len(gathered_metrics)
        avg_overlap_ratio = sum(m["overlap_ratio_pct"] for m in gathered_metrics) / len(gathered_metrics)

        print("\n" + "=" * 60)
        print(f" Benchmark Results ({dc_cfg.world_size} ranks, steps={dc_cfg.steps})")
        print("=" * 60)
        print(f" 实际物理通信耗时 (去重叠): {avg_comm:.3f} ms / step")
        print(f" 未遮蔽通信耗时 (Exposed): {avg_exposed:.3f} ms / step")
        print(f" 通信遮蔽率 (Overlap Ratio): {avg_overlap_ratio:.2f}%")
        print("=" * 60)

        results = {
            "summary": {
                "avg_actual_comm_per_step_ms": avg_comm,
                "avg_exposed_comm_per_step_ms": avg_exposed,
                "avg_overlap_ratio_pct": avg_overlap_ratio,
            },
            "per_rank_metrics": gathered_metrics,
        }
        with open(exp_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
        print(f"Artifacts and traces saved to: {exp_dir.resolve()}\n")

    dist.destroy_process_group()


def main(dc_cfg: DCConfig, model_cfg: ModelConfig) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = dc_cfg.save_base_dir / f"{timestamp}_{dc_cfg.world_size}_{dc_cfg.backend}_naive_ddp_cmcp_overlap"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # save config
    combined_args = {
        "dc_config": asdict(dc_cfg),
        "model_config": asdict(model_cfg),
    }
    with open(exp_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(combined_args, f, indent=4, default=str)

    mp.spawn(fn=run_benchmark, args=(dc_cfg, model_cfg, exp_dir), nprocs=dc_cfg.world_size, join=True)


def parse_args() -> tuple[DCConfig, ModelConfig]:
    parser = argparse.ArgumentParser(description="Naive DDP Communication-Computation Overlap Benchmark")
    setup_model_args(parser)
    setup_dc_benchmark_args(parser)
    args = parser.parse_args()

    model_cfg = create_model_config(args)
    dc_cfg = create_dc_config(args)
    return dc_cfg, model_cfg


if __name__ == "__main__":
    dc_cfg, model_cfg = parse_args()
    main(dc_cfg, model_cfg)
