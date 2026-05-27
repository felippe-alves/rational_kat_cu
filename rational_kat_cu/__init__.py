import torch
from kat_rational.rational_triton import rational_fwd_triton
from .rational_bwd_triton import rational_bwd_triton


class _RationalGroupedFn(torch.autograd.Function):
    """
    Grouped Padé rational activation.

    Forward:  Triton kernel (single fused launch, rational_fwd_triton).
    Backward: Triton kernel (rational_bwd_triton) with 2-D grid (g, blocks).
              Each block belongs to exactly one group, so tl.sum() collapses
              BLOCK contributions into a scalar before a single atomic_add per
              coefficient — reducing atomic contention by a factor of BLOCK
              compared to the element-wise approach in the original kernel.

    Polynomial convention (matches rational_triton.py):
        P(x) = a0 + a1*x + ... + a_m*x^m          (shared numerator)
        Q(x) = 1 + |b0|*|x| + ... + |b_{n-1}|*|x|^n  (per-group denominator)
        out  = P / Q
    """

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x, a_grouped, b, g):
        # x: (N, D)  a_grouped: (g, m+1)  b: (g, n)
        result = rational_fwd_triton(x, a_grouped.reshape(-1), b.reshape(-1), g)
        ctx.save_for_backward(x, a_grouped, b)
        ctx.g = g
        return result

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        x, a_grouped, b = ctx.saved_tensors
        g = ctx.g

        d_x, d_a, d_b = rational_bwd_triton(x, a_grouped, b, grad_output, g)

        return d_x, d_a, d_b, None   # None for g (non-tensor arg)


def _rat_cuda_impl(x, a, b):
    g = b.shape[0]
    a_grouped = a.unsqueeze(0).expand(g, -1).contiguous()
    return _RationalGroupedFn.apply(x, a_grouped, b, g)


# Disable torch.compile tracing so AOT autograd never tries to trace through
# our custom Triton kernels.  Both forward and backward run as opaque CUDA
# launches; torch.compile would otherwise try to re-compile the JIT-compiled
# Triton IR and may crash the pass manager on complex fused graphs.
try:
    rat_cuda = torch.compiler.disable(_rat_cuda_impl)
except AttributeError:
    rat_cuda = torch._dynamo.disable(_rat_cuda_impl)
