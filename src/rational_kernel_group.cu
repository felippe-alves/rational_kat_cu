#include <torch/extension.h>

template <typename scalar_t>
__global__ void rational_fwd_cuda_kernel_1dgroup(
    const scalar_t* __restrict__ x, 
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b, 
    scalar_t* __restrict__ result, 
    int B, int L, int D, int group, 
    int x_size, int D_per_group) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= x_size) return;

    // Calculate the index within the dimension D
    int d_index = idx % D;
    // Calculate the group index based on the position within dimension D
    int g_index = floor(d_index / D_per_group);

    // Calculate specific indices for a and b based on group
    int a_idx = g_index * 6;
    int b_idx = g_index * 4;

    // Load coefficients into registers (raw — abs applied after sum)
    scalar_t s_a[6], s_b[4];
    for (int i = 0; i < 6; ++i) {
        s_a[i] = a[a_idx + i];
    }
    for (int i = 0; i < 4; ++i) {
        s_b[i] = b[b_idx + i];
    }

    // Obtain the input value from the tensor
    scalar_t xp1 = x[idx];

    // Compute P(x) via Horner's method
    scalar_t P = s_a[5];
    for (int i = 4; i >= 0; --i) {
        P = fmaf(P, xp1, s_a[i]);
    }

    // Compute D(x) = b0*x + b1*x^2 + b2*x^3 + b3*x^4 via Horner
    scalar_t D = s_b[3];
    for (int i = 2; i >= 0; --i) {
        D = fmaf(D, xp1, s_b[i]);
    }
    D *= xp1;
    // Q(x) = 1 + |D(x)|
    scalar_t Q = 1.0f + fabsf(D);

    // Write the result of P / Q
    result[idx] = P / Q;
}

torch::Tensor rational_fwd_cuda_1dgroup(
    torch::Tensor x, 
    torch::Tensor n, 
    torch::Tensor d,
    int group
    ){
    auto result = at::empty_like(x);
    const int x_size = x.numel();
    int B = x.size(0);
    int L = x.size(1);
    int D = x.size(2);

    int threads_per_block = 256;
    int num_blocks = (x_size + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "rational_fwd_cuda_1dgroup", ([&] {
    rational_fwd_cuda_kernel_1dgroup<scalar_t>
        <<<num_blocks, threads_per_block>>>(
            x.data_ptr<scalar_t>(),
            n.data_ptr<scalar_t>(),
            d.data_ptr<scalar_t>(),
            result.data_ptr<scalar_t>(),
            B, L, D, group, x_size, D / group);
        }));

    return result;
}

// D(X) = b_0*X + b_1*X^2 + b_2*X^3 + b_3*X^4
// Q(X) = 1 + |D(X)|
// dQ/dx = sign(D) * D'(X)  where D'(X) = b_0 + 2*b_1*X + 3*b_2*X^2 + 4*b_3*X^3
// dF/dx = (-P/Q^2) * dQ/dx + dP/dx / Q
// dF/da_i = x^i / Q, i \in {0,5}
// dF/db_i = (-P/Q^2) * sign(D) * X^{i+1}, i \in {0,3}


template <typename scalar_t>
__global__ void rational_bwd_cuda_kernel_1dgroup(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    scalar_t* __restrict__ d_x,
    float* __restrict__ d_a,
    float* __restrict__ d_b,
    int B, int L, int D, int group, 
    int x_size, 
    const int n_size, 
    const int d_size,
    int D_per_group) {
    
    // Shared memory for accumulation
    // group < 32, so we can use 192 and 128 shared memory
    __shared__ float sda[192];
    __shared__ float sdb[128];
    // initialize shared memory to zero
    if ( threadIdx.x == 0) {
        for (int i = 0; i < 192; ++i) {
            sda[i] = 0;
        }
        for (int i = 0; i < 128; ++i) {
            sdb[i] = 0;
        }
    }

    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= x_size) return;

    // Calculate the index within the dimension D
    int d_index = idx % D;
    // Calculate the group index based on the position within dimension D
    int g_index = floor(d_index / D_per_group);

    // Calculate specific indices for a and b based on group
    int a_idx = g_index * 6;
    int b_idx = g_index * 4;

    // Load coefficients into registers (raw — abs applied after sum)
    scalar_t shared_a[6], shared_b[4];
    for (int i = 0; i < 6; ++i) {
        shared_a[i] = a[a_idx + i];
    }
    for (int i = 0; i < 4; ++i) {
        shared_b[i] = b[b_idx + i];
    }

    scalar_t local_da[6] = {0};
    scalar_t local_db[4] = {0};
    
    scalar_t xp = x[idx];
    // Compute powers of xp
    scalar_t xp_powers[5];
    xp_powers[0] = xp;
    xp_powers[1] = xp * xp_powers[0]; // xp^2
    xp_powers[2] = xp * xp_powers[1]; // xp^3
    xp_powers[3] = xp * xp_powers[2]; // xp^4
    xp_powers[4] = xp * xp_powers[3]; // xp^5

    scalar_t P = shared_a[0] 
    + shared_a[1] * xp_powers[0] 
    + shared_a[2] * xp_powers[1] 
    + shared_a[3] * xp_powers[2] 
    + shared_a[4] * xp_powers[3] 
    + shared_a[5] * xp_powers[4];

    // D(x) = b0*x + b1*x^2 + b2*x^3 + b3*x^4
    scalar_t D = shared_b[0] * xp_powers[0]
               + shared_b[1] * xp_powers[1]
               + shared_b[2] * xp_powers[2]
               + shared_b[3] * xp_powers[3];
    // Q(x) = 1 + |D(x)|
    scalar_t Q = scalar_t(1.0) + fabsf(D);

    scalar_t dP = shared_a[1] 
    + scalar_t(2.0) * shared_a[2] * xp_powers[0] 
    + scalar_t(3.0) * shared_a[3] * xp_powers[1] 
    + scalar_t(4.0) * shared_a[4] * xp_powers[2] 
    + scalar_t(5.0) * shared_a[5] * xp_powers[3];

    // D'(x) = b0 + 2*b1*x + 3*b2*x^2 + 4*b3*x^3
    scalar_t dD = shared_b[0]
    + scalar_t(2.0) * shared_b[1] * xp_powers[0]
    + scalar_t(3.0) * shared_b[2] * xp_powers[1]
    + scalar_t(4.0) * shared_b[3] * xp_powers[2];

    scalar_t sign_D = copysign(scalar_t(1.0), D);
    scalar_t dQ = sign_D * dD;

    scalar_t grad_o = grad_output[idx];
    
    scalar_t mpq2 = -P/(Q*Q);

    scalar_t d_i_x = (dP / Q + dQ * mpq2) * grad_o;
    d_x[idx] = d_i_x;

    // d_a contributions
    local_da[0] = scalar_t(1.0) / Q * grad_o;
    for (int i = 1; i < 6; ++i) {
        local_da[i] = (xp_powers[i-1] / Q) * grad_o;
    }

    // d_b contributions: -P/Q^2 * sign(D) * x^(i+1) * grad_o
    for (int i = 0; i < 4; ++i) {
        local_db[i] = mpq2 * sign_D * xp_powers[i] * grad_o;
    }

    // Reduce local arrays to shared memory
    for (int i = 0; i < 6; ++i) {
        atomicAdd(&sda[a_idx + i], local_da[i]);
    }
    for (int i = 0; i < 4; ++i) {
        atomicAdd(&sdb[b_idx + i], local_db[i]);
    }

    __syncthreads();

    // Only one thread writes back to global memory
    if (threadIdx.x == 0) {
        for (int i = 0; i < n_size; ++i) {
            atomicAdd(&d_a[i], sda[i]);
        }
        for (int i = 0; i < d_size; ++i) {
            atomicAdd(&d_b[i], sdb[i]);
        }
    }
}

std::vector<torch::Tensor> rational_bwd_cuda_1dgroup(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d, int group) {
    const int x_size = x.numel();
    const int n_size = n.numel();
    const int d_size = d.numel();

    auto d_x = at::empty_like(x);
    auto d_n = at::zeros_like(n).toType(at::kFloat);
    auto d_d = at::zeros_like(d).toType(at::kFloat);

    int B = x.size(0);
    int L = x.size(1);
    int D = x.size(2);

    int blockSize = 256;
    int numBlocks = (x_size + blockSize - 1) / blockSize;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "rational_bwd_cuda_1dgroup", ([&] {
    rational_bwd_cuda_kernel_1dgroup<scalar_t>
        <<<numBlocks, blockSize>>>(
            grad_output.data_ptr<scalar_t>(),
            x.data_ptr<scalar_t>(),
            n.data_ptr<scalar_t>(),
            d.data_ptr<scalar_t>(),
            d_x.data_ptr<scalar_t>(),
            d_n.data_ptr<float>(),
            d_d.data_ptr<float>(),
            B, L, D, group, x_size, n_size, d_size, D / group);
    }));

    return {d_x, d_n, d_d};
}