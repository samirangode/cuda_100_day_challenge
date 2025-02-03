#include <iostream>

__global__ void helloCUDA(){
    int threadId = threadIdx.x;
    printf("Hello from thread %d!\n", threadId);
}

int main(){
    helloCUDA<<<1, 10>>>();
    cudaDeviceSynchronize();
    return 0;
}