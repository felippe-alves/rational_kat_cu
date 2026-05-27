from kat_rational.rational_triton import RationalTriton1DGroup


def rat_cuda(x, a, b):
    """
    Triton-backed rational activation matching the nanochat/gpt.py calling convention.

    x: (N, d_in)   — 2D input (batch*seq flattened)
    a: (m+1,)      — shared numerator coefficients across all groups
    b: (g, n)      — per-group denominator coefficients

    Expands a to (g, m+1) then delegates to RationalTriton1DGroup.
    """
    g = b.shape[0]
    a_grouped = a.unsqueeze(0).expand(g, -1).contiguous()
    return RationalTriton1DGroup.apply(x, a_grouped, b, g)
