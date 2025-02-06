#include <iostream>
#include <cmath>

#define N 8 // Number of elements in a batch
#define EPS 1e-5 // Small number to prevent division by zero error

__global__ void layerNorm(float* input, float* output, float* mean, float* var, int width){
    __shared__ float shared_mean;
    __shared__ float shared_var;

    int tid = threadIdx.x;
    
    // Compute the mean only thread 0 will do it. Why???
    if(tid==0){
        float sum = 0.0;
        for(int i = 0; i<width; i++){
            sum+=input[i];
        }
        shared_mean = sum/width;
    }
    __syncthreads(); // Ensure all threads see updated shared_mean
    // Comput the varaince, again only thread 0 does it
    if(tid==0){
        float sum = 0.0;
        float val = 0.0;
        for(int i = 0; i<width; i++){
            val = (input[i] - shared_mean);
            sum+=val*val;
        }
        shared_var = sum/width;
    }
    __syncthreads();

    // Now we normalize each value
    output[tid] = (input[tid] - shared_mean) / sqrtf(shared_var + EPS);
}


int main(){
    int bytes = N * sizeof(float);

    // Allocate memory on the host
    float h_input[N] = {1,2,3,4,5,6,7,8};
    float h_output[N], h_mean, h_var;
    
    // Allocate memory on the device
    float* d_input, *d_output, *d_mean, *d_var;
    cudaMalloc(&d_input,bytes);
    cudaMalloc(&d_output,bytes);
    cudaMalloc(&d_mean, sizeof(float));
    cudaMalloc(&d_var, sizeof(float));


    //Copy data to device
    cudaMemcpy(d_input,h_input, bytes, cudaMemcpyHostToDevice);

    // Launch the kernel (1 block, N threads)
    layerNorm<<<1, N>>>(d_input, d_output, d_mean, d_var, N);

    //Copy result back to the host
    cudaMemcpy(h_output,d_output,bytes,cudaMemcpyDeviceToHost);

    std::cout << "Layer Normalized Output:\n";
    for (int i = 0; i < N; i++) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;
    
    // Free memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_mean);
    cudaFree(d_var);
}
