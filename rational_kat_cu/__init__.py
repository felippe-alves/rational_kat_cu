import torch
from kat_rational.rational_triton import rational_fwd_triton


class _RationalGroupedFn(torch.autograd.Function):
    """
    Grouped Padé rational activation.

    Forward: Triton kernel (fast, single fused launch per call).
    Backward: pure PyTorch — avoids the tl.atomic_add contention in the
    original Triton backward, where every thread hammered the same ~48
    coefficient gradient slots (100M → 48 atomic collisions on rat2).
    Here we recompute P/Q from saved x and reduce with .sum(), which
    dispatches efficient parallel CUDA reductions with no contention.

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
        g   = ctx.g
        N, D       = x.shape
        m_plus_1   = a_grouped.shape[1]   # e.g. 6
        n_coeff    = b.shape[1]           # e.g. 4
        D_per_group = D // g

        xg   = x.reshape(N, g, D_per_group)            # (N, g, Dg)
        grad = grad_output.reshape(N, g, D_per_group)   # (N, g, Dg)
        absx = xg.abs()

        b_abs = b.abs()   # (g, n)

        # Recompute Q via Horner on |x|
        Q = b_abs[:, -1].view(1, g, 1).expand(N, g, D_per_group)
        for i in range(n_coeff - 2, -1, -1):
            Q = Q * absx + b_abs[:, i].view(1, g, 1)
        Q = Q * absx + 1.0   # (N, g, Dg)

        # Recompute P via Horner on x
        P = a_grouped[:, -1].view(1, g, 1).expand(N, g, D_per_group)
        for i in range(m_plus_1 - 2, -1, -1):
            P = P * xg + a_grouped[:, i].view(1, g, 1)
        # P: (N, g, Dg)

        # dP/dx via Horner on derivative: a1 + x*(2a2 + x*(3a3 + ... + x*m*a_m))
        dP = (m_plus_1 - 1) * a_grouped[:, -1].view(1, g, 1).expand(N, g, D_per_group)
        for i in range(m_plus_1 - 2, 0, -1):
            dP = dP * xg + i * a_grouped[:, i].view(1, g, 1)
        # dP: (N, g, Dg)  — dP/dx of the numerator polynomial

        # dQ/d|x| = |b0| + 2|b1||x| + 3|b2||x|^2 + ...
        dQ_dabsx = b_abs[:, 0].view(1, g, 1).expand(N, g, D_per_group)
        absx_pow = absx
        for j in range(1, n_coeff):
            dQ_dabsx = dQ_dabsx + (j + 1) * b_abs[:, j].view(1, g, 1) * absx_pow
            absx_pow = absx_pow * absx
        dQ = xg.sign() * dQ_dabsx   # dQ/dx = sign(x) * dQ/d|x|

        Q2 = Q * Q

        # Gradient w.r.t. x: (dP/dx * Q - P * dQ/dx) / Q^2 * grad
        d_x = (dP / Q - P * dQ / Q2) * grad
        d_x = d_x.reshape(N, D)

        # Gradient w.r.t. a_grouped (shared numerator): sum over all groups and positions
        # d_a_grouped[g, i] = sum_{N, Dg} ( x^i / Q * grad )
        # PyTorch propagates this back through expand(g,-1) by summing over g dim,
        # giving the correct gradient for the original (m+1,) shaped a parameter.
        inv_Q_grad = grad / Q    # (N, g, Dg)
        d_a = a_grouped.new_zeros(g, m_plus_1)
        xpow = xg.new_ones(N, g, D_per_group)
        for i in range(m_plus_1):
            d_a[:, i] = (xpow * inv_Q_grad).sum(dim=(0, 2))
            if i < m_plus_1 - 1:
                xpow = xpow * xg

        # Gradient w.r.t. b (per-group denominator)
        # d_b[g, j] = sum_{N, Dg} ( -P/Q^2 * sign(b[g,j]) * |x|^(j+1) * grad )
        mpq2_grad = (-P / Q2) * grad    # (N, g, Dg)
        sign_b    = b.sign()            # (g, n)
        d_b       = b.new_zeros(g, n_coeff)
        absx_pow  = absx                # |x|^1
        for j in range(n_coeff):
            d_b[:, j] = (mpq2_grad * sign_b[:, j].view(1, g, 1) * absx_pow).sum(dim=(0, 2))
            absx_pow = absx_pow * absx

        return d_x, d_a, d_b, None   # None for g (non-tensor arg)


def rat_cuda(x, a, b):
    """
    Grouped Padé rational activation.

    x : (N, D)   — 2-D input (batch×seq flattened)
    a : (m+1,)   — shared numerator coefficients
    b : (g, n)   — per-group denominator coefficients
    """
    g = b.shape[0]
    a_grouped = a.unsqueeze(0).expand(g, -1).contiguous()
    return _RationalGroupedFn.apply(x, a_grouped, b, g)
