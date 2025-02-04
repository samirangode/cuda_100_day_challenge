// #include <iostream>

// // Kernel for vector addition __global__ defines a CUDA kernel
// __global__ void addVectors(int* a, int* b, int* c, int size){
//     int index = threadIdx.x + blockIdx.x + blockDim.x; // Unique thread ID
//     if(index<size){
//         c[index] = a[index] + b[index];
//     }
// }

// int main(){
//     int size = 10;
//     int bytes = size * sizeof(int);

//     // Allocate memory on host which is the CPU in this case
//     int h_a[size], h_b[size], h_c[size];

//     for(int i = 0; i< size; i++){
//         h_a[i] = i;
//         h_b[i] = i * 2;
//     }

//     // Allocate memory on the device GPU
//     int *d_a, *d_b, *d_c;
//     cudaMalloc(&d_a, bytes);
//     cudaMalloc(&d_b, bytes);
//     cudaMalloc(&d_c, bytes);

//     // Now copy data from the host to the device
//     cudaMemcpy(d_a, h_a, bytes,cudaMemcpyHostToDevice);
//     cudaMemcpy(d_b, h_b, bytes,cudaMemcpyHostToDevice);

//     // Launch Kernel
//     int threads_per_block = 256;
//     int blocks_needed = (size + threads_per_block - 1) / threads_per_block; // ceiling division
//     addVectors<<<blocks_needed, threads_per_block>>>(d_a, d_b, d_c, size);

//     cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

//     //print the result
//     std::cout<<"Vector addition result: \n";
//     for(int i = 0; i< size; i++) {
//         std::cout << h_c[i] << " ";
//     }
//     std::cout << std::endl;

//     // Free memory
//     cudaFree(d_a);
//     cudaFree(d_b);
//     cudaFree(d_c);

//     return 0;
// }

#include <iostream>

// CUDA Kernel for vector addition
__global__ void addVectors(int *a, int *b, int *c, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x; // Unique thread ID
    if (index < size) {
        c[index] = a[index] + b[index];
    }
}

int main() {
    int size = 10;
    int bytes = size * sizeof(int);

    // Allocate memory on host
    int h_a[size], h_b[size], h_c[size];
    for (int i = 0; i < size; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // Allocate memory on device
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // Launch kernel with 1 block and 10 threads
    addVectors<<<1, 10>>>(d_a, d_b, d_c, size);

    // Copy result back to host
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // Print result
    std::cout << "Vector addition result: ";
    for (int i = 0; i < size; i++) {
        std::cout << h_c[i] << " ";
    }
    std::cout << std::endl;

    // Free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}

