import json
import sys
from collections import defaultdict
from pathlib import Path

from cuda_memory_types import TraceEntry, load_snapshot

from cs336_systems.utils import MODEL_NAMES


def extract_peak_memory_gb(pickle_path: str, device_index: int = 0) -> float:
    snapshot = load_snapshot(pickle_path)
    device_traces = snapshot.get("device_traces", [])
    if device_index >= len(device_traces) or not device_traces[device_index]:
        return -1

    traces: list[TraceEntry] = device_traces[device_index]

    current_allocated: int = 0
    peak_allocated: int = 0

    for e in traces:
        action = e["action"]
        size = e.get("size", 0)

        if action == "alloc":
            current_allocated += size
            peak_allocated = max(peak_allocated, current_allocated)

        elif action == "free_requested":
            current_allocated = max(0, current_allocated - size)

    return peak_allocated / 1024**3


def collect_experiment_data(base_dir: str = "memory_res"):
    data = defaultdict(dict)

    for args_file in Path(base_dir).rglob("args.json"):
        pickle_file = args_file.with_name("memory_snapshot.pickle")
        if not pickle_file.exists():
            continue

        args = json.loads(args_file.read_text(encoding="utf-8"))
        m_cfg, b_cfg = args.get("model_cfg", {}), args.get("bench_cfg", {})
        raw_name = MODEL_NAMES.get(m_cfg.get("d_model"), "unknown")
        model_name = raw_name.upper() if raw_name in ("xl", "10b") else raw_name.capitalize()

        key = (
            model_name,
            b_cfg.get("mode", "train").capitalize(),
            m_cfg.get("context_length", 0),
        )
        tag = "mixedp" if b_cfg.get("use_mixed_precision") else "default"
        data[key][tag] = extract_peak_memory_gb(str(pickle_file))

    return dict(data)


def generate_typst(data):
    lines = [
        "#figure(",
        "  table(",
        "    columns: (auto, auto, auto, auto, auto, auto),",
        "    inset: (x: 8pt, y: 5pt),",
        "    align: (left, left, center, right, right, right),",
        "    stroke: none,",
        "    table.hline(stroke: 1.2pt),",
        "    table.header(",
        ("      [*Model*], [*Task*], [*Context*], [*FP32(GiB)*], [*Mixed BF16(GiB)*], [*Reduction*],"),
        "    ),",
        "    table.hline(stroke: 0.6pt),",
    ]

    sorted_keys = sorted(data.keys(), key=lambda x: (x[0], x[1], x[2]))
    prev_model = None

    for model, task, ctx in sorted_keys:
        if prev_model is not None and model != prev_model:
            lines.append("    table.hline(stroke: 0.4pt),")
        prev_model = model

        fp32 = data[(model, task, ctx)].get("default")
        mixed = data[(model, task, ctx)].get("mixedp")

        fp32_str = f"{fp32:.2f}" if fp32 is not None else "--"
        mixed_str = f"{mixed:.2f}" if mixed is not None else "--"

        if fp32 and mixed and fp32 > 0:
            diff = (mixed - fp32) / fp32 * 100
            diff_str = f"{diff:+.1f}%"
        else:
            diff_str = "--"

        lines.append(f"    [{model}], [{task}], [{ctx}], [{fp32_str}], [{mixed_str}], [{diff_str}],")

    lines.extend(
        [
            "    table.hline(stroke: 1.2pt),",
            "  ),",
            ("  caption: [Peak memory usage comparison with and without mixed precision.],"),
            ")\n",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    res_dir = sys.argv[1] if len(sys.argv) > 1 else "memory_res"

    if not Path(res_dir).exists():
        print(f"'{res_dir}' not found")
    else:
        results = collect_experiment_data(res_dir)
        print(generate_typst(results))
        print("\n" + "=" * 60 + "\n")
