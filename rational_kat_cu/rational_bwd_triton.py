"""
Triton backward kernel for the grouped Padé rational activation.

Polynomial convention (must match rational_triton.py forward):
    P(x) = a[0] + a[1]*x + ... + a[m]*x^m          (numerator, degree m=5, M1=6 coefficients)
    Q(x) = 1 + |b[0]|*|x| + ... + |b[n-1]|*|x|^n  (denominator, NC=4 terms)
    out  = P / Q

Grid is 2-D: (g, ceil(N*Dg / BLOCK))
  - axis-0 (pid_g) selects the group — so a[pid_g,:] and b[pid_g,:] are loaded
    as scalars and each block only ever touches one group's coefficient slots.
  - axis-1 (pid_b) tiles over the N*Dg elements that belong to that group.

Because every block in a given axis-0 row shares the same coefficient slots,
the atomic contention for d_a / d_b is reduced from
  (N*D / BLOCK) threads per slot  →  ceil(N*Dg / BLOCK) per slot,
a factor-of-g reduction.  More importantly, the per-block tl.sum() collapses
BLOCK contributions into a single scalar before any atomic, so the actual
atomic rate is ceil(N*Dg / BLOCK) — typically O(10^2) rather than O(10^5).
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------

@triton.jit
def _rational_bwd_kernel(
    x_ptr, a_ptr, b_ptr, grad_ptr,
    dx_ptr, da_ptr, db_ptr,
    N, D, Dg,
    M1: tl.constexpr,     # m+1 (number of numerator coefficients), e.g. 6
    NC: tl.constexpr,     # number of denominator coefficients, e.g. 4
    BLOCK: tl.constexpr,  # elements per block, e.g. 1024
):
    """
    Backward pass for grouped Padé rational activations.

    Each program instance handles:
      - pid_g : one group  (selects which a[g,:] / b[g,:] to load)
      - pid_b : a contiguous tile of N*Dg elements belonging to that group

    Gradients:
      dx[i]    = (dP/dx * Q - P * dQ/dx) / Q^2 * grad_out[i]
      d_a[g,k] += sum_block( x^k / Q * grad_out )
      d_b[g,j] += sum_block( -P/Q^2 * sign(b[g,j]) * |x|^(j+1) * grad_out )
    """
    pid_g = tl.program_id(0)   # group index
    pid_b = tl.program_id(1)   # block index within this group

    # Offsets within the (N * Dg) linear space for this group
    offs = pid_b * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N * Dg

    # Map flat group-local offset → global x offset
    # x is (N, D) stored row-major; within a group the elements are:
    #   row n, columns [pid_g*Dg, (pid_g+1)*Dg)
    n_idx  = offs // Dg                      # which row
    dg_idx = offs % Dg                       # which column within the group
    x_offs = n_idx * D + pid_g * Dg + dg_idx

    # Load input and upstream gradient
    x    = tl.load(x_ptr    + x_offs, mask=mask, other=0.0).to(tl.float32)
    grad = tl.load(grad_ptr + x_offs, mask=mask, other=0.0).to(tl.float32)

    # -----------------------------------------------------------------------
    # Load a[pid_g, :] and b[pid_g, :] as per-group scalars then broadcast
    # -----------------------------------------------------------------------
    a_base = pid_g * M1
    b_base = pid_g * NC

    # Numerator coefficients — unrolled at compile time via constexpr M1
    # We store them in a length-M1 array using tl.zeros + individual assignments.
    # Because M1 is constexpr Triton unrolls the loops below.

    a0 = tl.load(a_ptr + a_base + 0).to(tl.float32)
    a1 = tl.load(a_ptr + a_base + 1).to(tl.float32)
    a2 = tl.load(a_ptr + a_base + 2).to(tl.float32)
    a3 = tl.load(a_ptr + a_base + 3).to(tl.float32)
    a4 = tl.load(a_ptr + a_base + 4).to(tl.float32)
    a5 = tl.load(a_ptr + a_base + 5).to(tl.float32)

    b0 = tl.load(b_ptr + b_base + 0).to(tl.float32)
    b1 = tl.load(b_ptr + b_base + 1).to(tl.float32)
    b2 = tl.load(b_ptr + b_base + 2).to(tl.float32)
    b3 = tl.load(b_ptr + b_base + 3).to(tl.float32)

    b0_abs = tl.abs(b0)
    b1_abs = tl.abs(b1)
    b2_abs = tl.abs(b2)
    b3_abs = tl.abs(b3)

    abs_x = tl.abs(x)

    # -----------------------------------------------------------------------
    # Compute P(x) via Horner: a5*x^5 + ... + a0
    # -----------------------------------------------------------------------
    P = a5
    P = tl.fma(P, x, a4)
    P = tl.fma(P, x, a3)
    P = tl.fma(P, x, a2)
    P = tl.fma(P, x, a1)
    P = tl.fma(P, x, a0)

    # -----------------------------------------------------------------------
    # Compute Q(|x|) via Horner: 1 + |b0|*|x| + ... + |b3|*|x|^4
    # -----------------------------------------------------------------------
    Q = b3_abs
    Q = tl.fma(Q, abs_x, b2_abs)
    Q = tl.fma(Q, abs_x, b1_abs)
    Q = tl.fma(Q, abs_x, b0_abs)
    Q = tl.fma(Q, abs_x, 1.0)

    # -----------------------------------------------------------------------
    # Compute dP/dx via Horner: 5*a5*x^4 + 4*a4*x^3 + ... + a1
    # -----------------------------------------------------------------------
    dP = 5.0 * a5
    dP = tl.fma(dP, x, 4.0 * a4)
    dP = tl.fma(dP, x, 3.0 * a3)
    dP = tl.fma(dP, x, 2.0 * a2)
    dP = tl.fma(dP, x, a1)

    # -----------------------------------------------------------------------
    # Compute dQ/dx = sign(x) * dQ/d|x|
    # dQ/d|x| = |b0| + 2*|b1|*|x| + 3*|b2|*|x|^2 + 4*|b3|*|x|^3
    # -----------------------------------------------------------------------
    dQ_dabsx = 4.0 * b3_abs
    dQ_dabsx = tl.fma(dQ_dabsx, abs_x, 3.0 * b2_abs)
    dQ_dabsx = tl.fma(dQ_dabsx, abs_x, 2.0 * b1_abs)
    dQ_dabsx = tl.fma(dQ_dabsx, abs_x, b0_abs)

    sign_x = tl.where(x < 0.0, -1.0, 1.0)
    dQ = sign_x * dQ_dabsx

    Q2 = Q * Q

    # -----------------------------------------------------------------------
    # d_x (elementwise, stored directly)
    # -----------------------------------------------------------------------
    dx = (dP / Q - P * dQ / Q2) * grad
    tl.store(dx_ptr + x_offs, dx, mask=mask)

    # -----------------------------------------------------------------------
    # d_a: one atomic_add per coefficient per block
    # contrib = sum_block( x^k / Q * grad )
    # -----------------------------------------------------------------------
    inv_Q_grad = grad / Q

    # x powers
    xp1 = x
    xp2 = xp1 * x
    xp3 = xp2 * x
    xp4 = xp3 * x
    xp5 = xp4 * x

    ones = tl.full([BLOCK], 1.0, dtype=tl.float32)

    tl.atomic_add(da_ptr + a_base + 0, tl.sum(ones  * inv_Q_grad * mask, axis=0))
    tl.atomic_add(da_ptr + a_base + 1, tl.sum(xp1   * inv_Q_grad * mask, axis=0))
    tl.atomic_add(da_ptr + a_base + 2, tl.sum(xp2   * inv_Q_grad * mask, axis=0))
    tl.atomic_add(da_ptr + a_base + 3, tl.sum(xp3   * inv_Q_grad * mask, axis=0))
    tl.atomic_add(da_ptr + a_base + 4, tl.sum(xp4   * inv_Q_grad * mask, axis=0))
    tl.atomic_add(da_ptr + a_base + 5, tl.sum(xp5   * inv_Q_grad * mask, axis=0))

    # -----------------------------------------------------------------------
    # d_b: one atomic_add per coefficient per block
    # contrib = sum_block( -P/Q^2 * sign(b[j]) * |x|^(j+1) * grad )
    # -----------------------------------------------------------------------
    mpq2_grad = (-P / Q2) * grad

    sign_b0 = tl.where(b0 < 0.0, -1.0, 1.0)
    sign_b1 = tl.where(b1 < 0.0, -1.0, 1.0)
    sign_b2 = tl.where(b2 < 0.0, -1.0, 1.0)
    sign_b3 = tl.where(b3 < 0.0, -1.0, 1.0)

    axp1 = abs_x
    axp2 = axp1 * abs_x
    axp3 = axp2 * abs_x
    axp4 = axp3 * abs_x

    tl.atomic_add(db_ptr + b_base + 0, tl.sum(sign_b0 * axp1 * mpq2_grad * mask, axis=0))
    tl.atomic_add(db_ptr + b_base + 1, tl.sum(sign_b1 * axp2 * mpq2_grad * mask, axis=0))
    tl.atomic_add(db_ptr + b_base + 2, tl.sum(sign_b2 * axp3 * mpq2_grad * mask, axis=0))
    tl.atomic_add(db_ptr + b_base + 3, tl.sum(sign_b3 * axp4 * mpq2_grad * mask, axis=0))


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

def rational_bwd_triton(x, a_grouped, b, grad_output, g):
    """
    Compute gradients for the grouped Padé rational activation.

    Args:
        x           : (N, D)  input tensor, float32 on CUDA
        a_grouped   : (g, M1) numerator coefficients, float32 on CUDA
        b           : (g, NC) denominator coefficients, float32 on CUDA
        grad_output : (N, D)  upstream gradient, float32 on CUDA
        g           : int     number of groups

    Returns:
        dx  : (N, D)   gradient w.r.t. x
        da  : (g, M1)  gradient w.r.t. a_grouped
        db  : (g, NC)  gradient w.r.t. b
    """
    N, D = x.shape
    M1   = a_grouped.shape[1]   # m+1 = 6
    NC   = b.shape[1]           # n   = 4
    Dg   = D // g

    dx = torch.empty_like(x)
    da = torch.zeros_like(a_grouped)   # zeroed — kernel uses atomic_add
    db = torch.zeros_like(b)

    BLOCK = 1024
    grid  = (g, triton.cdiv(N * Dg, BLOCK))

    _rational_bwd_kernel[grid](
        x, a_grouped, b, grad_output,
        dx, da, db,
        N, D, Dg,
        M1=M1, NC=NC, BLOCK=BLOCK,
    )
    return dx, da, db


# ---------------------------------------------------------------------------
# Gradcheck test — run with:  python rational_bwd_triton.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    torch.manual_seed(0)
    N, D, g, m, n = 32, 128, 4, 5, 4
    M1 = m + 1   # 6
    Dg = D // g  # 32

    # Use float64 for gradcheck accuracy; the kernel runs in float32,
    # so we compare against a float32 PyTorch reference.
    device = "cuda"

    def pytorch_bwd(x, a_grouped, b, grad_output):
        """Pure-PyTorch reference backward (mirrors __init__.py logic)."""
        N, D = x.shape
        n_coeff = b.shape[1]
        m_plus_1 = a_grouped.shape[1]
        D_per_group = D // g

        xg   = x.reshape(N, g, D_per_group)
        grad = grad_output.reshape(N, g, D_per_group)
        absx = xg.abs()
        b_abs = b.abs()

        Q = b_abs[:, -1].view(1, g, 1).expand(N, g, D_per_group)
        for i in range(n_coeff - 2, -1, -1):
            Q = Q * absx + b_abs[:, i].view(1, g, 1)
        Q = Q * absx + 1.0

        P = a_grouped[:, -1].view(1, g, 1).expand(N, g, D_per_group)
        for i in range(m_plus_1 - 2, -1, -1):
            P = P * xg + a_grouped[:, i].view(1, g, 1)

        dP = (m_plus_1 - 1) * a_grouped[:, -1].view(1, g, 1).expand(N, g, D_per_group)
        for i in range(m_plus_1 - 2, 0, -1):
            dP = dP * xg + i * a_grouped[:, i].view(1, g, 1)

        dQ_dabsx = b_abs[:, 0].view(1, g, 1).expand(N, g, D_per_group)
        absx_pow = absx
        for j in range(1, n_coeff):
            dQ_dabsx = dQ_dabsx + (j + 1) * b_abs[:, j].view(1, g, 1) * absx_pow
            absx_pow = absx_pow * absx
        dQ = xg.sign() * dQ_dabsx

        Q2 = Q * Q
        d_x = (dP / Q - P * dQ / Q2) * grad
        d_x = d_x.reshape(N, D)

        inv_Q_grad = grad / Q
        d_a = a_grouped.new_zeros(g, m_plus_1)
        xpow = xg.new_ones(N, g, D_per_group)
        for i in range(m_plus_1):
            d_a[:, i] = (xpow * inv_Q_grad).sum(dim=(0, 2))
            if i < m_plus_1 - 1:
                xpow = xpow * xg

        mpq2_grad = (-P / Q2) * grad
        sign_b    = b.sign()
        d_b       = b.new_zeros(g, n_coeff)
        absx_pow  = absx
        for j in range(n_coeff):
            d_b[:, j] = (mpq2_grad * sign_b[:, j].view(1, g, 1) * absx_pow).sum(dim=(0, 2))
            absx_pow = absx_pow * absx

        return d_x, d_a, d_b

    # Random inputs and coefficients (float32 on CUDA)
    x_ref    = torch.randn(N, D, device=device, dtype=torch.float32)
    a_ref    = torch.randn(g, M1, device=device, dtype=torch.float32)
    b_ref    = torch.randn(g, n,  device=device, dtype=torch.float32)
    grad_ref = torch.randn(N, D, device=device, dtype=torch.float32)

    # PyTorch reference
    dx_ref, da_ref, db_ref = pytorch_bwd(x_ref, a_ref, b_ref, grad_ref)

    # Triton kernel
    dx_tri, da_tri, db_tri = rational_bwd_triton(x_ref, a_ref, b_ref, grad_ref, g)

    def check(name, ref, tri, atol=1e-4, rtol=1e-4):
        max_err = (ref - tri).abs().max().item()
        ok = torch.allclose(ref, tri, atol=atol, rtol=rtol)
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  {name:10s}  max_err={max_err:.2e}")
        return ok

    print("Gradcheck vs PyTorch reference:")
    all_ok  = check("dx",  dx_ref, dx_tri)
    all_ok &= check("da",  da_ref, da_tri)
    all_ok &= check("db",  db_ref, db_tri)

    if not all_ok:
        sys.exit(1)
    print("All checks passed.")
