import torch
import triton
from functools import partial
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def add_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    x_numel,
    x_block: gl.constexpr,
    y_block: gl.constexpr,
    layout: gl.constexpr,
):
    b_start = gl.program_id(0) * x_block + gl.program_id(1) * x_numel * y_block

    # Slice layouts needed here for arange.
    x_indices = gl.arange(0, x_block, layout=gl.SliceLayout(1, layout))
    y_indices = gl.arange(0, y_block, layout=gl.SliceLayout(0, layout))

    # Re-expand the layouts.
    offsets = b_start + x_indices[:, None] + y_indices[None, :] * x_numel

    a_ptrs = a_ptr + offsets
    b_ptrs = b_ptr + offsets

    a_v = gl.load(a_ptrs)
    b_v = gl.load(b_ptrs)

    c_ptrs = c_ptr + offsets

    gl.store(c_ptrs, (a_v + b_v))


def add_impl(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    x_block: int,
    y_block: int,
    layout,
    num_warps: int,
):
    ynumel = a.shape[0]
    xnumel = a.shape[1]
    grid = (
        triton.cdiv(xnumel, x_block),
        triton.cdiv(ynumel, y_block),
    )
    compiled_kernel = add_kernel[grid](
        a, b, c, xnumel, x_block, y_block, layout, num_warps=num_warps
    )
    return compiled_kernel


def bench_impl(impl, *args):
    compiled_kernel = impl(*args)
    fn = lambda: impl(*args)
    ms = triton.testing.do_bench(fn, warmup=10, rep=20, return_mode="mean")
    return ms


def bench():
    # Create inputs
    torch.manual_seed(0)
    xnumel = 2048
    ynumel = 2048
    a = torch.randn((xnumel, ynumel), device="cuda", dtype=torch.float32)
    b = torch.randn((xnumel, ynumel), device="cuda", dtype=torch.float32)

    # Run torch reference
    c_torch = torch.empty_like(a)

    def torch_impl(a, b, c):
        torch.add(a, b, out=c)

    torch_stats = bench_impl(torch_impl, a, b, c_torch)

    # Run Triton
    c_triton = torch.empty_like(a)
    triton_impl = partial(
        add_impl,
        x_block=128,
        y_block=32,
        layout=gl.BlockedLayout(
            size_per_thread=[2, 8],
            threads_per_warp=[32, 2],
            warps_per_cta=[2, 2],
            order=[0, 1],
        ),
        num_warps=4,
    )
    triton_stats = bench_impl(triton_impl, a, b, c_triton)

    # Get the numerical error
    err = torch.linalg.matrix_norm(c_torch - c_triton) / torch.linalg.matrix_norm(
        c_torch
    )

    print(f"Relative error: {err}")
    print(f"Torch stats: {torch_stats}")
    print(f"Triton stats: {triton_stats}")


if __name__ == "__main__":
    bench()
