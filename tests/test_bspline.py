"""Correctness tests for fused B-spline Triton kernels."""

import pytest
import torch


# ---------------------------------------------------------------------------
# Reference implementations (no external deps)
# ---------------------------------------------------------------------------

def _ref_b_splines(x, grid, spline_order):
    """Reference Cox-de Boor from kanprey/kan_layers.py."""
    x = x.unsqueeze(-1)
    bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
    for k in range(1, spline_order + 1):
        n = bases.shape[-1]
        t_i   = grid[:, :n - 1]
        t_ik  = grid[:, k:n - 1 + k]
        t_i1  = grid[:, 1:n]
        t_ik1 = grid[:, k + 1:n + k]
        left  = (x - t_i)   / (t_ik  - t_i ).clamp(min=1e-8) * bases[..., :-1]
        right = (t_ik1 - x) / (t_ik1 - t_i1).clamp(min=1e-8) * bases[..., 1:]
        bases = left + right
    return bases.contiguous()


def _make_grid(D, grid_size=8, spline_order=3, device="cuda"):
    step = 2.0 / grid_size
    grid_uniform = torch.linspace(-1, 1, grid_size + 1)
    left = grid_uniform[0] - step * torch.arange(spline_order, 0, -1)
    right = grid_uniform[-1] + step * torch.arange(1, spline_order + 1)
    full_grid = torch.cat([left, grid_uniform, right])
    return full_grid.unsqueeze(0).expand(D, -1).clone().to(device)


class _RefKANLinear(torch.nn.Module):
    """Minimal KANLinear for testing, matching efficient_kan API."""
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3):
        super().__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.n_basis = grid_size + spline_order
        self.base_weight = torch.nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.empty(out_features, in_features, self.n_basis))
        n_knots = grid_size + 2 * spline_order + 1
        grid_t = torch.linspace(-1, 1, grid_size + 1)
        step = 2.0 / grid_size
        left = grid_t[0] - step * torch.arange(spline_order, 0, -1)
        right = grid_t[-1] + step * torch.arange(1, spline_order + 1)
        grid_t = torch.cat([left, grid_t, right])
        self.register_buffer("grid", grid_t.unsqueeze(0).expand(in_features, -1).clone())
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=5**0.5)
        torch.nn.init.kaiming_uniform_(self.spline_weight, a=5**0.5)

    def b_splines(self, x):
        return _ref_b_splines(x, self.grid, self.spline_order)

    def forward(self, x):
        base = torch.nn.functional.linear(torch.nn.functional.silu(x), self.base_weight)
        spline = torch.einsum("bik,oik->bo", self.b_splines(x), self.spline_weight)
        return base + spline


# ---------------------------------------------------------------------------
# Basis evaluation tests
# ---------------------------------------------------------------------------

class TestBSplineBasis:
    @pytest.mark.parametrize("B,D", [(4, 32), (16, 128), (1, 64)])
    def test_forward_matches_reference(self, B, D):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        from rational_kat_cu.bspline_basis import bspline_basis_fwd

        torch.manual_seed(42)
        grid = _make_grid(D, device="cuda")
        x = torch.randn(B, D, device="cuda", dtype=torch.float32)

        out_triton = bspline_basis_fwd(x, grid, spline_order=3)
        out_ref = _ref_b_splines(x, grid, 3)

        max_err = (out_triton - out_ref).abs().max().item()
        assert max_err < 1e-5, f"Forward max error {max_err:.2e} exceeds 1e-5"
        assert out_triton.shape == (B, D, 11)

    @pytest.mark.parametrize("B,D,grid_size,order", [
        (4, 32, 5, 3), (4, 32, 8, 3), (4, 64, 3, 2),
    ])
    def test_forward_various_configs(self, B, D, grid_size, order):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        from rational_kat_cu.bspline_basis import bspline_basis_fwd

        n_knots = grid_size + 2 * order + 1
        if n_knots > 15:
            pytest.skip(f"n_knots={n_knots} exceeds kernel limit of 15")

        torch.manual_seed(42)
        grid = _make_grid(D, grid_size, order, device="cuda")
        x = torch.randn(B, D, device="cuda", dtype=torch.float32)
        out_triton = bspline_basis_fwd(x, grid, spline_order=order)
        out_ref = _ref_b_splines(x, grid, order)
        assert out_triton.shape == (B, D, grid_size + order)
        max_err = (out_triton - out_ref).abs().max().item()
        assert max_err < 1e-5, f"Max error {max_err:.2e}"

    def test_backward_matches_reference(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        from rational_kat_cu.bspline_basis import bspline_basis_bwd

        torch.manual_seed(42)
        B, D = 8, 64
        grid = _make_grid(D, device="cuda")
        x_ref = torch.randn(B, D, device="cuda", dtype=torch.float32, requires_grad=True)
        out_ref = _ref_b_splines(x_ref, grid, 3)
        grad_out = torch.randn_like(out_ref)
        out_ref.backward(grad_out)

        x_tri = x_ref.detach().requires_grad_(True)
        dx_triton = bspline_basis_bwd(x_tri, grid, grad_out.clone(), spline_order=3)
        max_err = (dx_triton - x_ref.grad).abs().max().item()
        assert max_err < 1e-4, f"Backward dx max error {max_err:.2e}"

    def test_gradcheck(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        from rational_kat_cu.bspline_basis import BSplineBasis

        torch.manual_seed(42)
        B, D = 4, 32
        grid = _make_grid(D, device="cuda")
        x = torch.randn(B, D, device="cuda", dtype=torch.float64, requires_grad=True)

        def fn(x):
            return BSplineBasis.apply(x.float(), grid.float(), 3).to(torch.float64)
        assert torch.autograd.gradcheck(fn, (x,), eps=1e-4, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# FusedKANLinear tests
# ---------------------------------------------------------------------------

class TestFusedKANLinear:
    def test_forward_shape(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        from rational_kat_cu.bspline_basis import FusedKANLinear

        torch.manual_seed(42)
        layer = FusedKANLinear(64, 128).cuda()
        x = torch.randn(4, 64, device="cuda")
        assert layer(x).shape == (4, 128)

    def test_forward_matches_reference(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        from rational_kat_cu.bspline_basis import FusedKANLinear

        torch.manual_seed(42)
        B, D = 4, 64
        ref = _RefKANLinear(D, D, grid_size=8).cuda()
        fused = FusedKANLinear(D, D, grid_size=8).cuda()
        fused.base_weight.data.copy_(ref.base_weight.data)
        fused.spline_weight.data.copy_(ref.spline_weight.data)
        fused.grid.copy_(ref.grid)

        x = torch.randn(B, D, device="cuda")
        max_err = (ref(x) - fused(x)).abs().max().item()
        assert max_err < 1e-5, f"Forward max error {max_err:.2e}"

    def test_backward_matches_reference(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        from rational_kat_cu.bspline_basis import FusedKANLinear

        torch.manual_seed(42)
        B, D = 4, 32
        ref = _RefKANLinear(D, D, grid_size=8).cuda()
        fused = FusedKANLinear(D, D, grid_size=8).cuda()
        fused.base_weight.data.copy_(ref.base_weight.data)
        fused.spline_weight.data.copy_(ref.spline_weight.data)
        fused.grid.copy_(ref.grid)

        x_ref = torch.randn(B, D, device="cuda", requires_grad=True)
        out_ref = ref(x_ref)
        grad_out = torch.randn_like(out_ref)
        out_ref.backward(grad_out)

        x_fused = x_ref.detach().clone().requires_grad_(True)
        out_fused = fused(x_fused)
        out_fused.backward(grad_out)

        for name, g_ref, g_fused in [
            ("dx", x_ref.grad, x_fused.grad),
            ("base_w", ref.base_weight.grad, fused.base_weight.grad),
            ("spline_w", ref.spline_weight.grad, fused.spline_weight.grad),
        ]:
            max_err = (g_ref - g_fused).abs().max().item()
            assert max_err < 1e-4, f"{name} max error {max_err:.2e}"

    def test_gradcheck(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        from rational_kat_cu.bspline_basis import FusedKANLinear

        torch.manual_seed(42)
        B, D = 2, 16
        layer = FusedKANLinear(D, D, grid_size=8).cuda().double()
        x = torch.randn(B, D, device="cuda", dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(layer, (x,), eps=1e-4, atol=1e-3, rtol=1e-3)

    def test_grid_update(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        from rational_kat_cu.bspline_basis import FusedKANLinear

        torch.manual_seed(42)
        layer = FusedKANLinear(32, 32).cuda()
        original = layer.grid.clone()
        x = torch.randn(64, 32, device="cuda")
        layer.update_grid(x)
        assert not torch.allclose(layer.grid, original), "Grid did not change"
