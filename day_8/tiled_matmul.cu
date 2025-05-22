#include <iostream>
#include <cstdlib>

#define TILE_SIZE 16
#define M 64
#define K 64
#define N 64

__global__ void matMulTiled(const float* A, const float* B, float* C, int m, int k, int n){
    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float val = 0.0f;

    for(int t= 0; t < (K + TILE_SIZE - 1)/TILE_SIZE; t++){
        // Load tiles into shared memory
        if(row<M && (t*TILE_SIZE + threadIdx.x)<K){
            s_A[threadIdx.y][threadIdx.x] = A[row*K + t*TILE_SIZE + threadIdx.x];
        }
        else{
            s_A[threadIdx.y][threadIdx.x] = 0.0f;   
        }

        if((t*TILE_SIZE + threadIdx.y)<K && col<N){
            s_B[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        }
        else{
            s_B[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();

        // Compute partial dot product for the tile
        for (int i = 0; i < TILE_SIZE; ++i)
            val += s_A[threadIdx.y][i] * s_B[i][threadIdx.x];
        
        __syncthreads();
    }

    if(row<M && col<N){
        C[row*N + col] = val;
    }
}


void initialize(float* mat, int rows, int cols){
    for(int i = 0; i<(rows*cols); i++){
        mat[i] = static_cast<float>(rand()%5 + 1);
    }
}


int main(){
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N *  sizeof(float);
    size_t size_C = M * N *  sizeof(float);

    float* h_A = (float*)malloc(size_A);
    float* h_B = (float*)malloc(size_B);
    float* h_C = (float*)malloc(size_C);

    initialize(h_A, M, K);
    initialize(h_B, K, N);

    float* d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1)/ TILE_SIZE, (M + TILE_SIZE - 1)/ TILE_SIZE );

    matMulTiled<<<blocks, threads>>>(d_A, d_B, d_C, M, K, N);

    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    std::cout<< "Done. Output Matrix C[0]: "<< h_C[0] << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}


