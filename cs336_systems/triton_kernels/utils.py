import functools

import torch
import triton


@functools.lru_cache(maxsize=16)
def get_hardware_limits(device: torch.device):
    """Query available shared memory budget for a single block on the target device."""
    props = torch.cuda.get_device_properties(device)
    max_smem_per_block = getattr(
        props,
        "max_shared_memory_per_block_optin",
        getattr(props, "max_shared_memory_per_block", 96 * 1024),
    )
    return int(max_smem_per_block * 0.90)


def get_optimal_tiles(
    d_model: int,
    seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
    num_stages: int = 2,
) -> tuple[int, int]:
    """
    Hardware-aware tile size selector considering Triton's pipelining num_stages.
    """
    usable_smem = get_hardware_limits(device)
    elem_bytes = torch.empty(0, dtype=dtype).element_size()

    # Candidate tiles in descending order of performance
    candidate_tiles = [
        (128, 128),
        (128, 64),
        (64, 64),
        (64, 32),
        (32, 32),
        (16, 16),
    ]

    best_q_tile = 32
    best_kv_tile = 32

    for q_tile, kv_tile in candidate_tiles:
        if q_tile > max(16, triton.next_power_of_2(seq_len)):
            continue

        # Keep Q in registers or a single SRAM buffer
        q_mem = q_tile * d_model * elem_bytes
        # K and V are pipelined and prefetched across loop iterations, occupying num_stages SRAM buffers
        kv_mem = num_stages * (2 * kv_tile * d_model * elem_bytes)

        # Add a reasonable amount of padding overhead for alignment (~1024 bytes)
        total_estimated_smem = q_mem + kv_mem + 1024

        if total_estimated_smem <= usable_smem:
            best_q_tile = q_tile
            best_kv_tile = kv_tile
            break

    return best_q_tile, best_kv_tile


def get_optimal_bwd_tiles(
    d_model: int,
    n_queries: int,
    n_keys: int,
    dtype: torch.dtype,
    device: torch.device,
    num_stages: int = 2,
) -> tuple[int, int]:
    """
    Hardware-aware tile size selector for backward kernels.
    Evaluates SRAM consumption limits across both flash_bwd_dq and flash_bwd_dkv kernels.
    """
    usable_smem = get_hardware_limits(device)
    elem_bytes = torch.empty(0, dtype=dtype).element_size()

    # Candidate tiles ordered by descending priority
    # Backward kernels exert higher SRAM/register pressure, typically starting from 64x64 downwards
    candidate_tiles = [
        (64, 64),
        (64, 32),
        (32, 64),
        (32, 32),
        (32, 16),
        (16, 16),
    ]

    best_q_tile = 32
    best_kv_tile = 32

    max_q_pow2 = max(16, triton.next_power_of_2(n_queries))
    max_k_pow2 = max(16, triton.next_power_of_2(n_keys))

    for q_tile, kv_tile in candidate_tiles:
        if q_tile > max_q_pow2 or kv_tile > max_k_pow2:
            continue

        # 1. Estimate SRAM footprint for flash_bwd_dq:
        # - Resident: Q, O, dO (each q_tile * d_model)
        # - Pipelined: K, V (each kv_tile * d_model across num_stages)
        # - Intermediate buffer: attention logits/scores (q_tile * kv_tile * 4 bytes for fp32)
        # - Alignment padding and metadata overhead (~2048 bytes)
        smem_dq = 3 * (q_tile * d_model * elem_bytes) + num_stages * (2 * kv_tile * d_model * elem_bytes) + (q_tile * kv_tile * 4) + 2048

        # 2. Estimate SRAM footprint for flash_bwd_dkv:
        # - Resident: K, V (each kv_tile * d_model)
        # - Pipelined: Q, dO (each q_tile * d_model across num_stages)
        # - Intermediate buffer: attention logits/scores (q_tile * kv_tile * 4 bytes for fp32)
        # - Alignment padding and metadata overhead (~2048 bytes)
        smem_dkv = 2 * (kv_tile * d_model * elem_bytes) + num_stages * (2 * q_tile * d_model * elem_bytes) + (q_tile * kv_tile * 4) + 2048

        # Both kernels must fit into the usable SRAM budget
        if max(smem_dq, smem_dkv) <= usable_smem:
            best_q_tile = q_tile
            best_kv_tile = kv_tile
            break

    return best_q_tile, best_kv_tile
