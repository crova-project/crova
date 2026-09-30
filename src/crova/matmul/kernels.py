"""Two Triton matmul kernels for out = left @ weight.T with a fixed reduction order.

ordered_tile  32x64 output tiles, K in blocks of 32 through tl.dot (tensor cores)
scalar        one output element per lane, a sequential FP32 multiply-add over K
"""
from collections.abc import Callable
from typing import Any

import triton
import triton.language as tl


def ordered_tile(
    left: Any, weight: Any, m: int, n: int, k: int, output_dtype: Any
) -> Callable[[], Any]:
    import torch

    @triton.jit
    def kernel(
        left_ptr,
        weight_ptr,
        output_ptr,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        block_m = tl.program_id(0)
        block_n = tl.program_id(1)
        offsets_m = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offsets_n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offsets_k = tl.arange(0, BLOCK_K)
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for block_k in range(0, tl.cdiv(K, BLOCK_K)):
            k_indices = block_k * BLOCK_K + offsets_k
            left_values = tl.load(
                left_ptr + offsets_m[:, None] * K + k_indices[None, :],
                mask=(offsets_m[:, None] < M) & (k_indices[None, :] < K),
                other=0.0,
            )
            weight_values = tl.load(
                weight_ptr + offsets_n[:, None] * K + k_indices[None, :],
                mask=(offsets_n[:, None] < N) & (k_indices[None, :] < K),
                other=0.0,
            )
            accumulator = tl.dot(left_values, tl.trans(weight_values), accumulator)
        tl.store(
            output_ptr + offsets_m[:, None] * N + offsets_n[None, :],
            accumulator,
            mask=(offsets_m[:, None] < M) & (offsets_n[None, :] < N),
        )

    def operation() -> Any:
        output = torch.empty((m, n), device=left.device, dtype=output_dtype)
        kernel[(triton.cdiv(m, 32), triton.cdiv(n, 64))](
            left,
            weight,
            output,
            M=m,
            N=n,
            K=k,
            BLOCK_M=32,
            BLOCK_N=64,
            BLOCK_K=32,
            num_warps=4,
            num_stages=2,
        )
        return output

    return operation


def scalar(
    left: Any, weight: Any, m: int, n: int, k: int, output_dtype: Any
) -> Callable[[], Any]:
    import torch

    @triton.jit
    def kernel(
        left_ptr,
        weight_ptr,
        output_ptr,
        OUTPUTS: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        BLOCK_OUTPUT: tl.constexpr,
    ):
        output_indices = tl.program_id(0) * BLOCK_OUTPUT + tl.arange(0, BLOCK_OUTPUT)
        valid = output_indices < OUTPUTS
        row_indices = output_indices // N
        column_indices = output_indices % N
        accumulator = tl.zeros((BLOCK_OUTPUT,), dtype=tl.float32)
        for reduction_index in tl.range(
            0,
            K,
            1,
            num_stages=1,
            loop_unroll_factor=1,
            disable_licm=True,
        ):
            left_values = tl.load(
                left_ptr + row_indices * K + reduction_index,
                mask=valid,
                other=0.0,
            ).to(tl.float32)
            weight_values = tl.load(
                weight_ptr + column_indices * K + reduction_index,
                mask=valid,
                other=0.0,
            ).to(tl.float32)
            accumulator += left_values * weight_values
        tl.store(output_ptr + output_indices, accumulator, mask=valid)

    def operation() -> Any:
        output = torch.empty((m, n), device=left.device, dtype=output_dtype)
        outputs = m * n
        kernel[(triton.cdiv(outputs, 128),)](
            left,
            weight,
            output,
            OUTPUTS=outputs,
            N=n,
            K=k,
            BLOCK_OUTPUT=128,
            num_warps=4,
            num_stages=1,
        )
        return output

    return operation

