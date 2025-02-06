#include <iostream>

__global__  void addMatrix(int* a, int* b, int* c, int width){
    int column = threadIdx.x + blockDim.x*blockIdx.x;
    int row = threadIdx.y + blockDim.y*blockIdx.y;

    if(row<width && column<width){
        int index = column*width + row;
        c[index] = a[index] + b[index] ;
    }
}

int main(){
    int N = 4;
    int size = N*N;
    int bytes = size * sizeof(int);

    // Allocate memory on the host
    int h_a[size], h_b[size], h_c[size];
    for(int i = 0; i<size; i++){
        h_a[i] = i;
        h_b[i] = i*2;
    }

    // Allocate memory on device
    int* d_a, *d_b, *d_c;
    cudaMalloc(&d_a,bytes);
    cudaMalloc(&d_b,bytes);
    cudaMalloc(&d_c,bytes);

    // Copy data to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // Define thread to device
    dim3 threadsPerBlock(2,2);
    dim3 blocksPerGrid((N+1)/2, (N+1)/2);

    // Launch Kernel
    addMatrix<<<blocksPerGrid, threadsPerBlock>>>(d_a,d_b,d_c,N);

    // Copy back to the host
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // Print result
    std::cout << "Matrix addition result:\n";
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << h_c[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    // Free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}

// #include <iostream>

// int main() {
//     int *d_a;
//     std::cout << "Before cudaMalloc, d_a = " << d_a << std::endl;

//     cudaMalloc(&d_a, 10 * sizeof(int));  

//     std::cout << "After cudaMalloc, d_a = " << d_a << std::endl;

//     cudaFree(d_a);  

//     std::cout << "After freeing, d_a = " << d_a << std::endl;

//     return 0;
// }
