import csv
import subprocess
import sys
from pathlib import Path

# 模型架构参数配置 (CS336 默认配置)
MODEL_CONFIGS = {
    "small": {"num_layers": 12, "d_model": 768, "num_heads": 12, "d_k": 64},
    "medium": {"num_layers": 24, "d_model": 1024, "num_heads": 16, "d_k": 64},
}


def is_gemm_kernel(kernel_name):
    name = kernel_name.lower()
    gemm_keywords = ["gemm", "mma", "cublas", "cutlass", "sm80_xmma"]
    return any(kw in name for kw in gemm_keywords)


def extract_att_insight(sqlite_path: Path):
    """提取 NVTX 标记的 QK^T、Softmax、AV 各阶段耗时"""
    cmd = ["nsys", "stats", "-r", "nvtx_kern_sum", "--format", "csv", "--timeunit", "ms", str(sqlite_path)]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        lines = res.stdout.splitlines()
    except Exception as e:
        print(f"[Error] 执行 nsys 失败: {sqlite_path}\n{e}")
        return None

    header_idx = -1
    for i, line in enumerate(lines):
        if "NVTX Range" in line and "Kernel Name" in line:
            header_idx = i
            break

    if header_idx == -1:
        return None

    reader = csv.DictReader(lines[header_idx:])

    score_gemm_ms = 0.0  # QK^T 的矩阵乘法
    final_matmul_ms = 0.0  # Softmax(S) * V 的矩阵乘法
    softmax_ms = 0.0  # Softmax 算子耗时
    softmax_calls = 0

    for row in reader:
        nvtx_range = row.get("NVTX Range", "").strip()
        k_name = row.get("Kernel Name", "").strip()
        try:
            total_ms = float(row.get("Total Time (ms)", 0.0))
            instances = int(row.get("Kern Inst", row.get("Instances", 0)))
        except ValueError:
            continue

        if not k_name or total_ms <= 0:
            continue

        # 1. Attention Scores: QK^T (过滤其中的矩阵乘法)
        if "computing attention scores" in nvtx_range:
            if is_gemm_kernel(k_name):
                score_gemm_ms += total_ms

        # 2. Softmax 操作
        elif "computing softmax" in nvtx_range:
            softmax_ms += total_ms
            softmax_calls += instances

        # 3. Final MatMul: Sftmx * V
        elif "final matmul" in nvtx_range:
            final_matmul_ms += total_ms

    return {
        "score_gemm_ms": score_gemm_ms,
        "final_matmul_ms": final_matmul_ms,
        "total_matmul_ms": score_gemm_ms + final_matmul_ms,
        "softmax_ms": softmax_ms,
        "softmax_calls": softmax_calls,
    }


def parse_folder_name(dir_name: str):
    # 形如: small_ctx512_att_insight
    parts = dir_name.split("_")
    model_type = parts[0]  # small 或 medium
    ctx_len = int(parts[1].replace("ctx", ""))
    return model_type, ctx_len


def calculate_theoretical_flops(model_type, ctx_len, batch_size, steps):
    """
    计算前向传播中所有 Attention 层的理论 FLOPs:
    1. QK^T: 2 * B * H * N^2 * d_k
    2. AV:   2 * B * H * N^2 * d_k
       -> MatMul Total = 4 * B * H * N^2 * d_k = 4 * B * N^2 * d_model
    3. Softmax: ~2 ~ 3 FLOPs / element (这里按标准约 2~3 FLOPs 计算，取 2.5)
       -> Softmax Total = 2.5 * B * H * N^2
    """
    cfg = MODEL_CONFIGS[model_type]
    num_layers = cfg["num_layers"]
    d_model = cfg["d_model"]
    num_heads = cfg["num_heads"]
    d_k = cfg["d_k"]
    N = ctx_len

    total_layers_called = num_layers * steps

    # 单层 FLOPs
    matmul_flops_per_layer = 4 * batch_size * num_heads * (N**2) * d_k
    softmax_flops_per_layer = 2.5 * batch_size * num_heads * (N**2)

    total_matmul_flops = matmul_flops_per_layer * total_layers_called
    total_softmax_flops = softmax_flops_per_layer * total_layers_called

    return total_matmul_flops, total_softmax_flops


def main():
    root_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 4

    att_dirs = sorted(root_dir.glob("*_att_insight"))
    if not att_dirs:
        print("[!] 未找到带有 *_att_insight 的目录！")
        return

    print("=" * 110)
    print(f"{'配置名称':<25} | {'MatMul耗时':<12} | {'Softmax耗时':<12} | {'耗时差距 (倍数)':<15} | {'FLOPs差距 (倍数)':<15}")
    print("=" * 110)

    for att_dir in att_dirs:
        sqlite_files = sorted(att_dir.glob("**/profile.sqlite"))
        if not sqlite_files:
            continue
        sqlite_path = sqlite_files[-1]

        model_type, ctx_len = parse_folder_name(att_dir.name)
        stats = extract_att_insight(sqlite_path)
        if not stats or stats["softmax_ms"] == 0:
            continue

        num_layers = MODEL_CONFIGS[model_type]["num_layers"]
        # 根据 Softmax 的调用次数估算 benchmark 的 steps
        steps = max(1, stats["softmax_calls"] // num_layers)

        total_matmul_ms = stats["total_matmul_ms"]
        softmax_ms = stats["softmax_ms"]

        # 计算理论 FLOPs
        matmul_flops, softmax_flops = calculate_theoretical_flops(model_type, ctx_len, batch_size, steps)

        # 吞吐与差距
        runtime_ratio = total_matmul_ms / softmax_ms if softmax_ms > 0 else 0
        flops_ratio = matmul_flops / softmax_flops if softmax_flops > 0 else 0

        matmul_tflops = (matmul_flops / (total_matmul_ms * 1e-3)) / 1e12 if total_matmul_ms > 0 else 0
        softmax_tflops = (softmax_flops / (softmax_ms * 1e-3)) / 1e12 if softmax_ms > 0 else 0

        config_tag = f"{model_type}_ctx{ctx_len}"
        print(f"{config_tag:<25} | {total_matmul_ms:>8.2f} ms   | {softmax_ms:>8.2f} ms   | {runtime_ratio:>8.2f} x         | {flops_ratio:>8.2f} x")

    print("=" * 110)
    print("\n[*] 深度解析 (含 TFLOP/s 吞吐量):")
    print("-" * 110)
    print(f"{'配置名称':<25} | {'MatMul 算力吞吐':<20} | {'Softmax 算力吞吐':<20} | {'算力吞吐差距':<15}")
    print("-" * 110)

    for att_dir in att_dirs:
        sqlite_files = sorted(att_dir.glob("**/profile.sqlite"))
        if not sqlite_files:
            continue
        model_type, ctx_len = parse_folder_name(att_dir.name)
        stats = extract_att_insight(sqlite_files[-1])
        if not stats or stats["softmax_ms"] == 0:
            continue

        num_layers = MODEL_CONFIGS[model_type]["num_layers"]
        steps = max(1, stats["softmax_calls"] // num_layers)

        matmul_flops, softmax_flops = calculate_theoretical_flops(model_type, ctx_len, batch_size, steps)
        matmul_tflops = (matmul_flops / (stats["total_matmul_ms"] * 1e-3)) / 1e12
        softmax_tflops = (softmax_flops / (stats["softmax_ms"] * 1e-3)) / 1e12
        tflops_ratio = matmul_tflops / softmax_tflops if softmax_tflops > 0 else 0

        config_tag = f"{model_type}_ctx{ctx_len}"
        print(f"{config_tag:<25} | {matmul_tflops:>10.2f} TFLOP/s    | {softmax_tflops:>10.4f} TFLOP/s    | {tflops_ratio:>8.1f} x")

    print("-" * 110)


if __name__ == "__main__":
    main()
