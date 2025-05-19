#include <iostream>

#define N 8 // Vector size
#define THREADS_PER_BLOCK 8 // Same as N for now

__global__ void parallelSum(float* input, float* output){
    __shared__ float shared_data[THREADS_PER_BLOCK]; //Shared memory

    int tid = threadIdx.x;

    // Load input into shared memory
    shared_data[tid] = input[tid];
    __syncthreads();
    
    // Perform parallel reduction
    for(int stride = THREADS_PER_BLOCK/2; stride>0; stride/=2){
        if(tid < stride) {
            shared_data[tid] += shared_data[tid+ stride];
        }
        __syncthreads(); // Wait for all threads
    }

    if(tid==0){
        *output = shared_data[0];
    }

}

int main(){
    int bytes = N * sizeof(float);

    // Allocate memory on host
    float h_input[N] = {1,2,3,4,5,6,7,8};
    float h_output;

    // Allocate memory on device
    float *d_input, *d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, sizeof(float));

    // Copy data to device
    cudaMemcpy(d_input,h_input,bytes,cudaMemcpyHostToDevice);

    // Launch kernel with 1 block and N threads
    parallelSum<<< 1, THREADS_PER_BLOCK>>>(d_input, d_output);

    // Copy result back to the host
    cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout<<h_output<<std::endl;

    cudaFree(d_input);
    cudaFree(d_output);
}