import csv
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


def run_nsys_and_extract(sqlite_path: Path):
    """调用 nsys 导出 CSV 到内存，并清洗归类"""
    cmd = ["nsys", "stats", "-r", "nvtx_kern_sum", "--format", "csv", "--timeunit", "ms", str(sqlite_path)]

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        lines = res.stdout.splitlines()
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"[Error] 执行 nsys 失败: {sqlite_path}")
        if isinstance(e, subprocess.CalledProcessError):
            print(f"stderr:\n{e.stderr}")
        else:
            print(e)
        return None, None

    header_idx = -1
    for i, line in enumerate(lines):
        if "NVTX Range" in line and "Kernel Name" in line:
            header_idx = i
            break

    if header_idx == -1:
        return None, None

    reader = csv.DictReader(lines[header_idx:])
    rows = list(reader)

    # 1. 动态获取主线程 TID
    main_tid = None
    for row in rows:
        nvtx_range = row.get("NVTX Range", "").strip()
        if nvtx_range.endswith(":forward") or nvtx_range == "forward":
            main_tid = row.get("TID", "").strip()
            break

    forward_kernels = defaultdict(lambda: {"count": 0, "total_ms": 0.0})
    backward_kernels = defaultdict(lambda: {"count": 0, "total_ms": 0.0})

    # 2. 严格按阶段归类 (排除 optimizer/prepare 及主线程 cuBLAS 嵌套)
    for row in rows:
        nvtx_range = row.get("NVTX Range", "").strip()
        tid = row.get("TID", "").strip()
        k_name = row.get("Kernel Name", "").strip()

        try:
            total_ms = float(row.get("Total Time (ms)", 0.0))
            count = int(row.get("Kern Inst", 0))
        except ValueError:
            continue

        if not k_name or total_ms <= 0:
            continue

        # Forward 桶
        if tid == main_tid and (nvtx_range.endswith(":forward") or nvtx_range == "forward"):
            forward_kernels[k_name]["count"] += count
            forward_kernels[k_name]["total_ms"] += total_ms

        # Backward 桶 (主线程 backward 清理 + 子线程发射的所有算子)
        elif (tid == main_tid and ":backward" in nvtx_range) or (tid != main_tid and tid != ""):
            backward_kernels[k_name]["count"] += count
            backward_kernels[k_name]["total_ms"] += total_ms

    # 3. 按耗时降序排序
    sorted_fwd = sorted(forward_kernels.items(), key=lambda x: x[1]["total_ms"], reverse=True)
    sorted_bwd = sorted(backward_kernels.items(), key=lambda x: x[1]["total_ms"], reverse=True)

    return sorted_fwd, sorted_bwd


def print_table(data_list, title, top_k=10):
    """打印格式化表格"""
    total_phase_ms = sum(d["total_ms"] for _, d in data_list)
    print(f"\n--- {title} (Total: {total_phase_ms:.2f} ms) ---")
    print(f"{'Rank':<5} {'Time (ms)':<12} {'Ratio':<8} {'Calls':<8} {'Avg (ms)':<10} Kernel Name")
    print("-" * 100)

    display_items = data_list[:top_k] if top_k else data_list
    for rank, (name, d) in enumerate(display_items, 1):
        ratio = (d["total_ms"] / total_phase_ms * 100) if total_phase_ms > 0 else 0
        avg_ms = d["total_ms"] / d["count"] if d["count"] else 0
        short_name = name[:60] + "..." if len(name) > 60 else name
        print(f"{rank:<5} {d['total_ms']:<12.4f} {ratio:>5.1f}%  {d['count']:<8} {avg_ms:<10.4f} {short_name}")

    if top_k and len(data_list) > top_k:
        print(f"... 剩余 {len(data_list) - top_k} 个底层细小 Kernel 已省略 (详见导出的 CSV 文件)")


def export_to_csv(output_path: Path, fwd_list, bwd_list):
    """将清洗后的纯净数据导出为 CSV"""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Phase", "Kernel Name", "Total Time (ms)", "Calls", "Avg Time (ms)"])
        for name, d in fwd_list:
            avg_ms = d["total_ms"] / d["count"] if d["count"] else 0
            writer.writerow(["Forward", name, f"{d['total_ms']:.4f}", d["count"], f"{avg_ms:.4f}"])
        for name, d in bwd_list:
            avg_ms = d["total_ms"] / d["count"] if d["count"] else 0
            writer.writerow(["Backward", name, f"{d['total_ms']:.4f}", d["count"], f"{avg_ms:.4f}"])


def main():
    root_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 10  # 终端默认展示前 10 个

    sqlite_files = sorted(root_dir.glob("**/profile.sqlite"))
    if not sqlite_files:
        print(f"[Error] 在 {root_dir} 下未找到任何 profile.sqlite 文件！")
        return

    print(f"[*] 找到 {len(sqlite_files)} 个实验目录，开始分类排序...\n")

    for path in sqlite_files:
        parts = path.parts
        config_name = parts[-3] if len(parts) >= 3 else path.parent.name
        timestamp = parts[-2] if len(parts) >= 3 else ""

        print("\n" + "=" * 105)
        print(f"  实验配置: [{config_name}]  (时间戳: {timestamp})")
        print("=" * 105)

        sorted_fwd, sorted_bwd = run_nsys_and_extract(path)
        if sorted_fwd is None or sorted_bwd is None:
            continue

        # 1. 终端打印 Forward
        print_table(sorted_fwd, "【FORWARD PASS】Kernel 耗时排名", top_k=top_k)

        # 2. 终端打印 Backward
        print_table(sorted_bwd, "【BACKWARD PASS】Kernel 耗时排名", top_k=top_k)

        # 3. 自动在每个实验目录下保存一份清晰完整的 CSV
        clean_csv_path = path.parent / "clean_kernels.csv"
        export_to_csv(clean_csv_path, sorted_fwd, sorted_bwd)
        print(f"\n[OK] 完整无截断的清洗数据已保存至: {clean_csv_path}")


if __name__ == "__main__":
    main()
