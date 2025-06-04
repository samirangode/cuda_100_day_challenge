#include <iostream>
#include <cmath>

#define N 4
#define D 4
#define TILE 2

__global__ void tiled_attention(const float* Q, const float* K, const float* V, float* O) {
    __shared__ float Qs[TILE][D];
    __shared__ float Ks[TILE][D];
    __shared__ float Vs[TILE][D];

    int tile_row = blockIdx.x;   // which tile of rows we're working on
    int tile_col = threadIdx.x;  // which thread in the block

    // Each block processes TILE rows of Q at a time
    int q_row_start = tile_row * TILE;

    for (int i = 0; i < TILE; ++i) {
        int q_row = q_row_start + i;
        if (q_row < N && tile_col < D)
            Qs[i][tile_col] = Q[q_row * D + tile_col];
    }
    __syncthreads();

    // Loop over K, V tiles along the sequence dimension
    float partial_O[TILE][D] = {0};
    for (int k_tile = 0; k_tile < (N + TILE - 1) / TILE; ++k_tile) {
        // Load K, V tile into shared memory
        for (int i = 0; i < TILE; ++i) {
            int kv_row = k_tile * TILE + i;
            if (kv_row < N && tile_col < D) {
                Ks[i][tile_col] = K[kv_row * D + tile_col];
                Vs[i][tile_col] = V[kv_row * D + tile_col];
            }
        }
        __syncthreads();

        // For each Q row in this tile:
        for (int qi = 0; qi < TILE; ++qi) {
            if (q_row_start + qi < N) {
                float attn_scores[TILE];
                float max_score = -INFINITY;
                for (int ki = 0; ki < TILE; ++ki) {
                    float score = 0.0f;
                    for (int d = 0; d < D; ++d)
                        score += Qs[qi][d] * Ks[ki][d];
                    attn_scores[ki] = score;
                    if (score > max_score) max_score = score;
                }
                // Softmax for the current Q row over this tile of K
                float exp_sum = 0.0f;
                for (int ki = 0; ki < TILE; ++ki) {
                    attn_scores[ki] = expf(attn_scores[ki] - max_score);
                    exp_sum += attn_scores[ki];
                }
                for (int ki = 0; ki < TILE; ++ki)
                    attn_scores[ki] /= exp_sum + 1e-8f;

                // Accumulate partial output (for each D)
                for (int d = 0; d < D; ++d)
                    for (int ki = 0; ki < TILE; ++ki)
                        partial_O[qi][d] += attn_scores[ki] * Vs[ki][d];
            }
        }
        __syncthreads();
    }

    // Write output
    for (int qi = 0; qi < TILE; ++qi) {
        int out_row = q_row_start + qi;
        if (out_row < N && tile_col < D)
            O[out_row * D + tile_col] = partial_O[qi][tile_col];
    }
}

int main() {
    float h_Q[N * D] = {1, 2, 3, 4,  5, 6, 7, 8,  9,10,11,12, 13,14,15,16};
    float h_K[N * D] = {1, 0, 0, 0,  0, 1, 0, 0,  0, 0, 1, 0,  0, 0, 0, 1};
    float h_V[N * D] = {1, 2, 3, 4,  5, 6, 7, 8,  9,10,11,12, 13,14,15,16};
    float h_O[N * D] = {0};

    float *d_Q, *d_K, *d_V, *d_O;
    cudaMalloc(&d_Q, N * D * sizeof(float));
    cudaMalloc(&d_K, N * D * sizeof(float));
    cudaMalloc(&d_V, N * D * sizeof(float));
    cudaMalloc(&d_O, N * D * sizeof(float));

    cudaMemcpy(d_Q, h_Q, N * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, N * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, N * D * sizeof(float), cudaMemcpyHostToDevice);

    tiled_attention<<<N / TILE, D>>>(d_Q, d_K, d_V, d_O);

    cudaMemcpy(h_O, d_O, N * D * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Tiled Attention Output:" << std::endl;
    for (int i = 0; i < N; ++i) {
        for (int d = 0; d < D; ++d)
            std::cout << h_O[i * D + d] << " ";
        std::cout << std::endl;
    }

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
    return 0;
}
