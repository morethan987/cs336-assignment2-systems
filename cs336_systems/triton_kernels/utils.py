import functools

import torch
import triton


@functools.lru_cache(maxsize=16)
def get_hardware_limits(device: torch.device):
    """Query available shared memory budget for the target device."""
    props = torch.cuda.get_device_properties(device)
    max_smem_per_block = getattr(
        props,
        "max_shared_memory_per_multiprocessor",
        96 * 1024,
    )
    # Reserve a safety headroom to prevent exceeding hardware limits
    usable_smem = min(max_smem_per_block * 0.75, 96 * 1024)
    return usable_smem


def get_optimal_tiles(d_model: int, seq_len: int, dtype: torch.dtype, device: torch.device) -> tuple[int, int]:
    """
    Hardware-aware tile size selector returning only (Q_TILE_SIZE, KV_TILE_SIZE).
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
        # Avoid tiles larger than sequence length
        if q_tile > max(16, triton.next_power_of_2(seq_len)):
            continue

        # Estimated SRAM = (Q + 2*K/V)*d_model*bytes + intermediate attention (Q_TILE * KV_TILE * 4)
        tensors_mem = (q_tile + 2 * kv_tile) * d_model * elem_bytes
        inter_mem = q_tile * kv_tile * 4
        if tensors_mem + inter_mem <= usable_smem:
            best_q_tile = q_tile
            best_kv_tile = kv_tile
            break

    return best_q_tile, best_kv_tile
