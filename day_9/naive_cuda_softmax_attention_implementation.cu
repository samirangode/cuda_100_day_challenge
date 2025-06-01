// #include <iostream>
// #include <cmath>

// #define N 4
// #define D 2

// __global__ void naive_attention(const float* Q, const float* K, const float* V, float* O){

//     int i = threadIdx.x;

//     float scores[N];
//     for(int j = 0; j<N; j++){
//         float score = 0.0f;
//         for(int d = 0; d<D; d++){
//             score += Q[i*D + d] * K[j*D + d];
//         }
//         scores[j] = score;
//     }

//     // Now calculate the softmax for the given row
//     float max_scores = scores[0];
//     for(int j = 1; j<D; j++){
//         max_scores = max(max_scores, scores[j]);
//     }
//     float exp_sum = 0.0f;
//     for(int j = 0; j<N; j++){
//         scores[j] = expf(scores[j] - max_scores);
//         exp_sum+=scores[j];
//     }
    
//     for(int j = 0; j<N; j++){
//         scores[j]/=exp_sum;
//     }

//     for(int d=0; d<D; d++){
//         float out = 0.0f;
//         for(int j = 0; j<N; j++){
//             out+=scores[j] * V[j*D + d];
//         }
//         O[i*D + d] = out;
//     }
// }

// int main(){
//     float h_Q[N*D] = {};
//     float h_K[N*D] = {};
//     float h_V[N*D] = {};
//     float h_O[N*D];

//     float *d_Q, *d_K, *d_V, *d_O;
//     cudaMalloc(&d_Q,N*D*sizeof(float));
//     cudaMalloc(&d_K,N*D*sizeof(float));
//     cudaMalloc(&d_V,N*D*sizeof(float));
//     cudaMalloc(&d_O,N*D*sizeof(float));

//     cudaMemcpy(d_Q, h_Q, N*D*sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_K, h_K, N*D*sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_V, h_V, N*D*sizeof(float), cudaMemcpyHostToDevice);

//     naive_attention<<<1, N>>>(d_Q, d_K, d_V, d_O);

//     cudaMemcpy(h_O, d_O, N*D*sizeof(float), cudaMemcpyDeviceToHost);

//     std::cout << "Naive Attention Output:" << std::endl;

//     for (int i = 0; i < N; ++i) {
//         for (int d = 0; d < D; ++d)
//             std::cout << h_O[i * D + d] << " ";
//         std::cout << std::endl;
//     }

//     cudaFree(d_Q); 
//     cudaFree(d_K);
//     cudaFree(d_V);
//     cudaFree(d_O);

//     return 0;
// }

#include <iostream>
#include <cmath>

#define N 4   // sequence length
#define D 2   // embedding size

__global__ void naive_attention(const float* Q, const float* K, const float* V, float* O) {
    int i = threadIdx.x;  // row in Q, S, O

    // Compute attention scores S[i][j] = dot(Q[i], K[j])
    float scores[N];
    for (int j = 0; j < N; ++j) {
        float score = 0.0f;
        for (int d = 0; d < D; ++d)
            score += Q[i * D + d] * K[j * D + d];
        scores[j] = score;
    }

    // Softmax over scores
    float max_score = scores[0];
    for (int j = 1; j < N; ++j)
        if (scores[j] > max_score) max_score = scores[j];
    float exp_sum = 0.0f;
    for (int j = 0; j < N; ++j) {
        scores[j] = expf(scores[j] - max_score);
        exp_sum += scores[j];
    }
    for (int j = 0; j < N; ++j)
        scores[j] /= exp_sum;

    // Output O[i] = sum_j softmax(S[i][j]) * V[j]
    for (int d = 0; d < D; ++d) {
        float out = 0.0f;
        for (int j = 0; j < N; ++j)
            out += scores[j] * V[j * D + d];
        O[i * D + d] = out;
    }
}

int main() {
    float h_Q[N * D] = {1, 0, 0, 1, 1, 1, 0, 0};
    float h_K[N * D] = {1, 0, 0, 1, 1, 1, 0, 0};
    float h_V[N * D] = {1, 2, 3, 4, 5, 6, 7, 8};
    float h_O[N * D];

    float *d_Q, *d_K, *d_V, *d_O;
    cudaMalloc(&d_Q, N * D * sizeof(float));
    cudaMalloc(&d_K, N * D * sizeof(float));
    cudaMalloc(&d_V, N * D * sizeof(float));
    cudaMalloc(&d_O, N * D * sizeof(float));

    cudaMemcpy(d_Q, h_Q, N * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, N * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, N * D * sizeof(float), cudaMemcpyHostToDevice);

    naive_attention<<<1, N>>>(d_Q, d_K, d_V, d_O);

    cudaMemcpy(h_O, d_O, N * D * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Naive Attention Output:" << std::endl;
    for (int i = 0; i < N; ++i) {
        for (int d = 0; d < D; ++d)
            std::cout << h_O[i * D + d] << " ";
        std::cout << std::endl;
    }

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
    return 0;
}
