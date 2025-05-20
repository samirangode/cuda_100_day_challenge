#include <iostream>
#include <cmath>

#define N 32
#define FULL_MASK 0xFFFFFFFF

__inline__ __device__ float warpReduceMax(float val){
    for(int offset = 16; offset>0; offset/=2){
        val = fmaxf(val, __shfl_xor_sync(FULL_MASK, val, offset));
    }
    return val;
}

__inline__ __device__ float warpReduceSum(float val){
    for(int offset = 16; offset>0; offset/=2){
        val += __shfl_xor_sync(FULL_MASK, val, offset);
    }
    return val;
}

__global__ void softmaxWarp(float* input, float* output){
    int tid = threadIdx.x;
    float x = input[tid];
    
    float max_val = warpReduceMax(x);
    float exp_x = expf(x-max_val);
    float sum_exp = warpReduceSum(exp_x);
    output[tid] = exp_x / sum_exp;
}

int main(){
    float h_input[N];
    for(int i = 0; i<N; i++){
        h_input[i] = i+1;
    }

    float h_output[N];

    float* d_input;
    float* d_output;

    cudaMalloc(&d_input, N*sizeof(float));
    cudaMalloc(&d_output, N*sizeof(float));

    cudaMemcpy(d_input, h_input, N*sizeof(float), cudaMemcpyHostToDevice);

    softmaxWarp<<<1, N>>>(d_input, d_output);

    cudaMemcpy(d_output, h_output, N*sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Softmax Output" << std::endl;

    for(int i = 0; i<N; i++){
        std::cout<<h_output[i]<<" ";
    }
    std::cout<<std::endl;

    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
