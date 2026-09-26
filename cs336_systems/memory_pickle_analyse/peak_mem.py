import sys
from pathlib import Path

from cuda_memory_types import TraceEntry, load_snapshot


def extract_peak_memory_gb(pickle_path: Path, device_index: int = 0) -> float:
    snapshot = load_snapshot(str(pickle_path))

    # 1. Calculate active allocated memory at snapshot dump time (baseline)
    actual_allocated_at_dump = 0
    for seg in snapshot.get("segments", []):
        if seg.get("device", 0) == device_index:
            for b in seg.get("blocks", []):
                if b.get("state") == "active_allocated":
                    actual_allocated_at_dump += b.get("size", 0)

    device_traces = snapshot.get("device_traces", [])
    if device_index >= len(device_traces) or not device_traces[device_index]:
        return actual_allocated_at_dump / 1024**3 if actual_allocated_at_dump > 0 else -1.0

    traces: list[TraceEntry] = device_traces[device_index]

    # 2. Track dynamic memory allocation changes within recorded traces
    current_allocated: int = 0
    peak_delta: int = 0

    for e in traces:
        action = e["action"]
        size = e.get("size", 0)
        if action == "alloc":
            current_allocated += size
            peak_delta = max(peak_delta, current_allocated)
        elif action == "free_requested":
            current_allocated = max(0, current_allocated - size)

    # 3. Compensate: memory present at dump exceeding trace end represents unrecorded static memory
    base_allocated = max(0, actual_allocated_at_dump - current_allocated)
    return (base_allocated + peak_delta) / 1024**3


def main():
    res_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "memory_res")
    if not res_dir.exists():
        print(f"Directory '{res_dir}' not found.")
        return

    # Collect and sort all snapshot files by path
    snapshot_files = sorted(res_dir.rglob("memory_snapshot.pickle"))
    if not snapshot_files:
        print(f"No 'memory_snapshot.pickle' found in '{res_dir}'.")
        return

    # Extract peak memory for each experiment folder
    results = []
    for p in snapshot_files:
        folder_name = p.parent.relative_to(res_dir).as_posix()
        peak_gb = extract_peak_memory_gb(p)
        results.append((folder_name, peak_gb))

    max_len = max(len(name) for name, _ in results)

    # Print aligned results
    print(f"\n{'Experiment Folder':<{max_len}} | Peak Memory")
    print("-" * (max_len + 15))
    for folder_name, peak_gb in results:
        val_str = f"{peak_gb:.2f} GiB" if peak_gb >= 0 else "Error / Empty"
        print(f"{folder_name:<{max_len}} | {val_str:>10}")
    print()


if __name__ == "__main__":
    main()
