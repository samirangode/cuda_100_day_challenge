#include <iostream>
#include <cmath>

#define N 4
#define D 4
#define TILE 2

__global__ void conv2d_kernel(const float* img,
                              const float* kernel,
                              float* output,
                              int H, int W, int K
){
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.x;
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


int main(){
    int H = 64, W = 64, K = 4;
    int outH = H-K+1, outW = W-K+1;

    float img[H*W];
    float kernel[K*K];
    float output[outH*outW];

    float val = 0.0;
    for(int i = 0; i<H; i++){
        for(int j = 0; j<W; j++){
            img[i*H + j] = val;
            val+=1;
        }
    }
    




    // Launching CUDA Kernel
    dim3 blockDim(16, 16);
    dim3 gridDim((outW + blockDim.x - 1)/blockDim.x,
                 (outH + blockDim.y - 1)/blockDim.y);
    conv2d_kernel<<<gridDim, blockDim>>>(img, kernel, output, H, W, K);

    return 0;
}
