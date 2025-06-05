#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

#define TILE_SIZE 16


__global__ void matmul_tiled(
    const float* __restrict__ A, const float* __restrict__ B, float* C,
    int M, int K, int N
)
{
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float acc = 0.0f;
    for(int t = 0; t<(K + TILE_SIZE - 1)/TILE_SIZE; t++){
        
        int tiled_col = t*TILE_SIZE + threadIdx.x;
        int tiled_row = t*TILE_SIZE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = 
            (row < M && tiled_col < K) ? A[row * K + tiled_col] : 0.0f;
        
        Bs[threadIdx.y][threadIdx.x] = 
            (col < N && tiled_row < K) ? B[tiled_row * N + col] : 0.0f;
        __syncthreads();

        for(int k = 0; k<TILE_SIZE; k++){
            acc+=As[threadIdx.y][k]*Bs[k][threadIdx.x];
        }
        __syncthreads();
    }
    
    if(row<M && col<N)
        C[row*N + col] = acc;
}

void initialize(float* mat, int rows, int col){
    for(int i = 0; i<rows*col; i++){
        mat[i] = static_cast<float>(rand()%5 + 1);
    }
}

int main(){
    int M = 50, K = 78, N = 63;

    size_t size_B = M * K * sizeof(float);
    size_t size_A = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float* h_A  = (float*)malloc(size_A);
    float* h_B  = (float*)malloc(size_B);
    float* h_C  = (float*)malloc(size_C);

    initialize(h_A, M, K);
    initialize(h_B, K, N);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    // C is of shape M x N, so M rows and N cols, that is, M is on y axis
    // and N is on the x axis 
    dim3 blocks((N + TILE_SIZE -1)/TILE_SIZE, (M + TILE_SIZE -1)/TILE_SIZE);

    matmul_tiled<<<blocks, threads>>>(d_A, d_B, d_C, M, K, N);
