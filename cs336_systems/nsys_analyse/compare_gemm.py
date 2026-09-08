import csv
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


def is_gemm_kernel(kernel_name):
    name = kernel_name.lower()
    gemm_keywords = ["gemm", "mma", "cublas", "cutlass", "sm80_xmma"]
    return any(kw in name for kw in gemm_keywords)


def extract_forward_nvtx(sqlite_path: Path):
    """从 forward_only 的 sqlite 中，仅提取 NVTX 标记为 'forward' 的 Kernel"""
    cmd = ["nsys", "stats", "-r", "nvtx_kern_sum", "--format", "csv", "--timeunit", "ms", str(sqlite_path)]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        lines = res.stdout.splitlines()
    except Exception as e:
        print(f"[Error] 解析 NVTX 失败: {sqlite_path}\n{e}")
        return None

    header_idx = -1
    for i, line in enumerate(lines):
        if "NVTX Range" in line and "Kernel Name" in line:
            header_idx = i
            break

    if header_idx == -1:
        return None

    reader = csv.DictReader(lines[header_idx:])
    total_ms = 0.0
    gemm_ms = 0.0
    other_kernels = defaultdict(float)

    for row in reader:
        nvtx_range = row.get("NVTX Range", "").strip()
        k_name = row.get("Kernel Name", "").strip()

        # 核心过滤：只提取处于 forward 范围内的 Kernel
        if "forward" not in nvtx_range.lower():
            continue

        try:
            t_ms = float(row.get("Total Time (ms)", 0.0))
        except ValueError:
            continue

        if not k_name or t_ms <= 0:
            continue

        total_ms += t_ms
        if is_gemm_kernel(k_name):
            gemm_ms += t_ms
        else:
            other_kernels[k_name] += t_ms

    sorted_others = sorted(other_kernels.items(), key=lambda x: x[1], reverse=True)
    return total_ms, gemm_ms, sorted_others


def extract_train_step_full(sqlite_path: Path):
    """从 train_step 的 sqlite 中提取整个训练步（包含 fwd, bwd, opt）的所有 Kernel"""
    cmd = ["nsys", "stats", "-r", "cuda_gpu_kern_sum", "--format", "csv", "--timeunit", "ms", str(sqlite_path)]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        lines = res.stdout.splitlines()
    except Exception as e:
        print(f"[Error] 解析 TrainStep 失败: {sqlite_path}\n{e}")
        return None

    header_idx = -1
    for i, line in enumerate(lines):
        if "Total Time (ms)" in line and "Name" in line:
            header_idx = i
            break

    if header_idx == -1:
        return None

    reader = csv.DictReader(lines[header_idx:])
    total_ms = 0.0
    gemm_ms = 0.0
    other_kernels = defaultdict(float)

    for row in reader:
        k_name = row.get("Name", "").strip()
        try:
            t_ms = float(row.get("Total Time (ms)", 0.0))
        except ValueError:
            continue

        if not k_name or t_ms <= 0:
            continue

        total_ms += t_ms
        if is_gemm_kernel(k_name):
            gemm_ms += t_ms
        else:
            other_kernels[k_name] += t_ms

    sorted_others = sorted(other_kernels.items(), key=lambda x: x[1], reverse=True)
    return total_ms, gemm_ms, sorted_others


def main():
    root_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")

    # 获取所有实验组目录
    fwd_dirs = sorted(root_dir.glob("*_forward_only"))

    print("[*] 开始基于 NVTX 提取纯净的 Forward 与完整 TrainStep 对比数据...\n")

    for fwd_dir in fwd_dirs:
        base_name = fwd_dir.name.replace("_forward_only", "")
        train_dir = root_dir / f"{base_name}_train_step"

        # 找到两边最新的 sqlite
        fwd_sqlites = sorted(fwd_dir.glob("**/profile.sqlite"))
        train_sqlites = sorted(train_dir.glob("**/profile.sqlite"))

        if not fwd_sqlites or not train_sqlites:
            continue

        fwd_sqlite = fwd_sqlites[-1]
        train_sqlite = train_sqlites[-1]

        # 1. 用 NVTX 提取纯 Forward 数据
        fwd_stats = extract_forward_nvtx(fwd_sqlite)
        # 2. 提取完整 TrainStep 数据
        train_stats = extract_train_step_full(train_sqlite)

        if not fwd_stats or not train_stats:
            continue

        fwd_total, fwd_gemm, fwd_others = fwd_stats
        train_total, train_gemm, train_others = train_stats

        fwd_gemm_ratio = (fwd_gemm / fwd_total * 100) if fwd_total > 0 else 0
        train_gemm_ratio = (train_gemm / train_total * 100) if train_total > 0 else 0

        print("=" * 100)
        print(f"【模型配置】: {base_name}")
        print("=" * 100)

        # 打印 Forward Pass 结果
        print("[FORWARD ONLY (纯前向推理, 基于 NVTX 过滤)]")
        print(f"  -> 总耗时: {fwd_total:.2f} ms")
        print(f"  -> GEMM 耗时: {fwd_gemm:.2f} ms (占比: {fwd_gemm_ratio:.1f}%)")
        print(f"  -> Non-GEMM 耗时: {fwd_total - fwd_gemm:.2f} ms (占比: {100 - fwd_gemm_ratio:.1f}%)")
        print("  -> 前 3 大其他 Kernel:")
        for k_name, k_ms in fwd_others[:3]:
            print(f"     * {(k_ms / fwd_total * 100):>5.1f}% | {k_ms:>6.2f} ms | {k_name[:65]}")

        # 打印 Train Step 结果
        print("\n[COMPLETE TRAIN STEP (前向 + 反向 + 优化器)]")
        print(f"  -> 总耗时: {train_total:.2f} ms (约为 Forward 的 {train_total / fwd_total:.2f} 倍)")
        print(f"  -> GEMM 耗时: {train_gemm:.2f} ms (占比: {train_gemm_ratio:.1f}%)")
        print(f"  -> Non-GEMM 耗时: {train_total - train_gemm:.2f} ms (占比: {100 - train_gemm_ratio:.1f}%)")
        print("  -> 前 3 大其他 Kernel:")
        for k_name, k_ms in train_others[:3]:
            print(f"     * {(k_ms / train_total * 100):>5.1f}% | {k_ms:>6.2f} ms | {k_name[:65]}")

        print("\n")


if __name__ == "__main__":
    main()
