"""Correctness tests for the grouped Pade rational Triton kernels.

Compares Triton forward/backward against a pure-PyTorch reference implementation
that uses the canonical formula: Q(x) = 1 + |b0*x + b1*x^2 + b2*x^3 + b3*x^4|.
"""

import pytest
import torch
from rational_kat_cu import rat_cuda


# ---------------------------------------------------------------------------
# PyTorch reference implementation
# ---------------------------------------------------------------------------

def _rat_ref(x, a, b):
    """Reference: same formula as Triton kernel and gpt.py Horner fallback.

    x: (N, D)
    a: (M1,) shared numerator coefficients
    b: (G, NC) per-group denominator coefficients
    """
    G, Dg = b.shape[0], x.shape[-1] // b.shape[0]
    M1 = a.shape[0]
    NC = b.shape[1]
    x_g = x.reshape(-1, G, Dg)  # (N, G, Dg)

    # P(x) via Horner
    num = a[-1]
    for i in range(M1 - 2, -1, -1):
        num = a[i] + x_g * num

    # D(x) = b0*x + b1*x^2 + ... via Horner
    d = b[:, -1].view(1, G, 1)
    for i in range(NC - 2, -1, -1):
        d = b[:, i].view(1, G, 1) + x_g * d
    denom = 1.0 + (x_g * d).abs()

    return (num / denom).reshape(x.shape)


# ---------------------------------------------------------------------------
# Forward correctness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("N,D,G", [
    (4, 64, 8),
    (32, 256, 8),
    (128, 768, 8),
    (1, 32, 4),
])
def test_forward_correctness(N, D, G):
    """Triton forward must match PyTorch reference within 1e-5."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(42)
    x = torch.randn(N, D, device='cuda', dtype=torch.float32)
    a = torch.randn(6, device='cuda', dtype=torch.float32)
    b = torch.randn(G, 4, device='cuda', dtype=torch.float32)

    out_triton = rat_cuda(x, a, b)
    out_ref = _rat_ref(x, a, b)

    assert out_triton.shape == x.shape
    max_err = (out_triton - out_ref).abs().max().item()
    assert max_err < 1e-5, f"Forward max error {max_err:.2e} exceeds 1e-5"


# ---------------------------------------------------------------------------
# Backward correctness via gradcheck
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("N,D,G", [
    (4, 64, 1),
    (8, 256, 1),
    (4, 64, 4),
    (8, 128, 8),
])
def test_gradcheck(N, D, G):
    """torch.autograd.gradcheck passes for 1 and multiple groups."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(42)
    x = torch.randn(N, D, device='cuda', dtype=torch.float64, requires_grad=True)
    a = torch.randn(6, device='cuda', dtype=torch.float64, requires_grad=True)
    b = torch.randn(G, 4, device='cuda', dtype=torch.float64, requires_grad=True)

    assert torch.autograd.gradcheck(
        rat_cuda, (x, a, b),
        eps=1e-4, atol=1e-3, rtol=1e-3,
        raise_exception=True,
    )


# ---------------------------------------------------------------------------
# Gradient comparison against reference
# ---------------------------------------------------------------------------

def test_backward_vs_reference():
    """Triton backward gradients match PyTorch reference autograd."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(42)
    N, D, G = 32, 256, 8
    x = torch.randn(N, D, device='cuda', dtype=torch.float32)
    a = torch.randn(6, device='cuda', dtype=torch.float32)
    b = torch.randn(G, 4, device='cuda', dtype=torch.float32)

    # Reference via autograd
    x_ref = x.clone().detach().requires_grad_(True)
    a_ref = a.clone().detach().requires_grad_(True)
    b_ref = b.clone().detach().requires_grad_(True)
    out_ref = _rat_ref(x_ref, a_ref, b_ref)
    grad_out = torch.randn_like(out_ref)
    out_ref.backward(grad_out)

    # Triton
    x_tri = x.clone().detach().requires_grad_(True)
    a_tri = a.clone().detach().requires_grad_(True)
    b_tri = b.clone().detach().requires_grad_(True)
    out_tri = rat_cuda(x_tri, a_tri, b_tri)
    out_tri.backward(grad_out)

    for name, g_ref, g_tri in [
        ("dx", x_ref.grad, x_tri.grad),
        ("da", a_ref.grad, a_tri.grad),
        ("db", b_ref.grad, b_tri.grad),
    ]:
        max_err = (g_ref - g_tri).abs().max().item()
        assert max_err < 1e-4, f"{name} max error {max_err:.2e} exceeds 1e-4"


# ---------------------------------------------------------------------------
# BF16 autocast correctness
# ---------------------------------------------------------------------------

def test_bf16_autocast():
    """Forward with bf16 inputs must match fp32 reference (autocast cast_inputs)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 not supported on this GPU")

    torch.manual_seed(42)
    N, D, G = 16, 128, 8
    x_f32 = torch.randn(N, D, device='cuda', dtype=torch.float32)
    a_f32 = torch.randn(6, device='cuda', dtype=torch.float32)
    b_f32 = torch.randn(G, 4, device='cuda', dtype=torch.float32)

    x_bf16 = x_f32.to(torch.bfloat16)
    a_bf16 = a_f32.to(torch.bfloat16)
    b_bf16 = b_f32.to(torch.bfloat16)

    out_f32 = rat_cuda(x_f32, a_f32, b_f32)
    out_bf16 = rat_cuda(x_bf16, a_bf16, b_bf16)

    max_err = (out_f32 - out_bf16.float()).abs().max().item()
    assert max_err < 5e-3, f"BF16 autocast max error {max_err:.2e} exceeds 5e-3"


# ---------------------------------------------------------------------------
# Identity initialization produces near-identity output
# ---------------------------------------------------------------------------

def test_identity_init():
    """rat1 with identity coefficients: output equals input."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(42)
    N, D, G = 8, 128, 8
    x = torch.randn(N, D, device='cuda', dtype=torch.float32)

    a = torch.zeros(6, device='cuda', dtype=torch.float32)
    a[1] = 1.0  # a1=1, others=0 -> P(x)=x
    b = torch.zeros(G, 4, device='cuda', dtype=torch.float32)  # D=0 -> Q=1

    out = rat_cuda(x, a, b)
    max_err = (out - x).abs().max().item()
    assert max_err < 1e-5, f"Identity init max error {max_err:.2e} exceeds 1e-5"
