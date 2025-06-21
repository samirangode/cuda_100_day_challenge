#include <iostream>
#include <cmath>

#define N 4
#define D 4
#define TILE 16

__global__ void conv2d_kernel(const float* img,
                              const float* kernel,
                              float* output,
                              int H, int W, int K
){
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int outH = H - K + 1, outW = W - K + 1;

    if(out_x < outW && out_y < outH){
        float sum = 0.0f;
        for(int i = 0; i<K; i++){
            for(int j = 0; j<K; j++){
                int img_x = out_x + j;
                int img_y = out_y + i;
                sum += img[img_y * W + img_x] * kernel[i*K + j];
            }
        }
        output[out_y * outW + out_x] = sum;
    }

}

__global__ void conv2d_shared(const float* img, const float* kernel, float* output,
                              int H, int W, int K, int P, int S)
{
    extern __shared__ float sh_img[]; // dynamically allocated shared memory

    int tx = threadIdx.x; int ty = threadIdx.y;
    int out_x = blockIdx.x * blockDim.x + tx;
    int out_y = blockIdx.y * blockDim.y + ty;
    int outH = (H + 2 * P - K) / S + 1;
    int outW = (W + 2 * P - K) / S + 1;

    // Calculate shared memory patch coordinates (with halo from the filter)
    int sh_x = tx;
    int sh_y = ty;
    int img_x = out_x * S - P; 
    int img_y = out_y * S - P;
    // Load input into shared memory
    if(img_x >= 0 && img_x < W && img_y >= 0 && img_y < H){
        sh_img[sh_y * (TILE + K - 1) + sh_x] = img[img_y * W + img_x];
    }
    else{
        sh_img[sh_y * (TILE + K - 1) + sh_x] = 0.0f;
    }
    __syncthreads();

    // compute within bounds
    if(out_x<outW && out_y<outH){
        float sum = 0.0f;
        for(int i = 0; i<K; i++){
            for(int j = 0; j<K; j++){
                sum += sh_img[(sh_y + i) * (TILE + K - 1) + (sh_x + j)] * kernel[ i * K + j];
            }
        }
        output[out_y * outW + out_x] = sum;
    }

}

__global__ void conv2d_kernel_pad_stride(
    const float* img, const float* kernel, float* output,
    int H, int W, int K, int P, int S
){
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    int outH = (H + 2 * P - K) / S + 1;
    int outW = (W + 2 * P - K) / S + 1;

    if(out_x < outW && out_y < outH){
        float sum = 0.0f;
        for(int i = 0; i < K; i++){
            for(int j = 0; j < K; j++){
                int img_x = out_x * S + j - P;
                int img_y = out_y * S + j - P;

                if(img_x >= 0 && img_x < W && img_y >= 0 && img_y < H)
                    sum += img[img_y * W + img_x] * kernel[i * K + j];
            }
        }
        output[out_y * outW + out_x] = sum;
    }


}


int main(){
    int H = 64, W = 64, K = 4, P = 2, S = 2;
    int outH = (H + 2 * P - K) / S + 1;
    int outW = (W + 2 * P - K) / S + 1;
 

    float img[H*W];
    float kernel[K*K];
    float output[outH*outW];

    float val = 1.0;
    for(int i = 0; i<H; i++){
        for(int j = 0; j<W; j++){
            img[i*W + j] = val;
            val+=1;
        }
    }
    val = 1.0;
    for(int i = 0; i<K; i++){
        for(int j = 0; j<K; j++){
            kernel[i*K + j] = val;
            val+=1;
        }
    }

    float* img_device, * kernel_device, *output_device;
    cudaMalloc(&img_device, H * W * sizeof(float));
    cudaMalloc(&kernel_device, K * K * sizeof(float));
    cudaMalloc(&output_device, outH * outW * sizeof(float));

    // transfer
    cudaMemcpy(img_device, img, H * W * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(kernel_device, kernel, K * K * sizeof(float), cudaMemcpyHostToDevice);


    // Launching CUDA Kernel
    dim3 blockDim(16, 16);
    dim3 gridDim((outW + blockDim.x - 1)/blockDim.x,
                 (outH + blockDim.y - 1)/blockDim.y);
    conv2d_kernel<<<gridDim, blockDim>>>(img_device, kernel_device, output_device, H, W, K);

    cudaMemcpy(output, output_device, outH * outW * sizeof(float), cudaMemcpyDeviceToHost);

    for(int i = 0; i<outH; i++){
        for(int j = 0; j<outW; j++){
            std::cout<<output[i*outW + j]<<" ";
        }
        std::cout<<std::endl;
    }

    cudaFree(img_device); cudaFree(kernel_device); cudaFree(output_device);

    return 0;
}
