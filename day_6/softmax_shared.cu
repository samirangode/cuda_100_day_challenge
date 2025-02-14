#include <iostream>
#include <cmath>

#define N 8 // Softmax input size (must be a power of 2)
#define THREADS_PER_BLOCK 8 // equal to N

__global__ void softMaxShared(float* input, float* output){

    __shared__ float shared_data[N]; // shared memory for intermediate values

    int tid = threadIdx.x;

    // load data into shared memory
    shared_data[tid] = expf(input[tid]);
    __syncthreads(); // so that everyone loads it and there are no race conditions

    // comput sum of exponentials using parallel reduction
    for(int stride = N/2; stride>0; stride=stride/2){
        if(tid<stride/2){
            shared_data[tid] = shared_data[tid] + shared_data[tid + stride];
        }
        __syncthreads();
    }

}