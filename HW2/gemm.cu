/*
GEMM implementation in CUDA
*/

#include <iostream>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>

const int TILE_SIZE = 16;

__global__ void gemm_kernel(float *A, float *B, float *C, int M, int N, int K) {

    // Shared memory for tiles
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    // Calculate row and column index of the element in C
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float value = 0;

    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load data into shared memory
        if (t * TILE_SIZE + threadIdx.x < K && row < M) {
            tile_A[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (t * TILE_SIZE + threadIdx.y < K && col < N) {
            tile_B[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads(); // Wait for all threads to load the tiles

        // Perform the multiplication
        for (int i = 0; i < TILE_SIZE; ++i) {
            value += tile_A[threadIdx.y][i] * tile_B[i][threadIdx.x];
        }

        __syncthreads(); // Wait for all threads to complete the multiplication
    }

    // Write the result to C
    if (row < M && col < N) {
        atomicAdd(&C[row * N + col], value); // Use atomic addition to handle concurrent writes
    }
}


void gemm_cuda(float *A, float *B, float *C, int M, int N, int K) {
    float *d_A, *d_B, *d_C;

    // Allocate device memory
    cudaMalloc((void**)&d_A, M * K * sizeof(float));
    cudaMalloc((void**)&d_B, K * N * sizeof(float));
    cudaMalloc((void**)&d_C, M * N * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, M * N * sizeof(float)); // Initialize C to 0

    // Define grid and block dimensions
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    // Launch the kernel
    gemm_kernel<<<grid, threads>>>(d_A, d_B, d_C, M, N, K);

    // Copy result from device to host
    cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}


void gemm_cpu(float *A, float *B, float *C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i * N + j] = 0.0f;
            for (int k = 0; k < K; ++k) {
                C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
}


int main(int argc, char **argv){
    int M, N, K;
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <M> <N> <K>" << std::endl;
        return 1;
    }

    M = atoi(argv[1]);
    N = atoi(argv[2]);
    K = atoi(argv[3]);

    float *A = new float[M * K];
    float *B = new float[K * N];
    float *C_cpu = new float[M * N];
    float *C_cuda = new float[M * N];
    // Initialize matrices A and B with random values
    for (int i = 0; i < M * K; i++) {
        A[i] = (rand() % 11) - 5;
    }
    for (int i = 0; i < K * N; i++) {
        B[i] = (rand() % 11) - 5;
    }

    // Compute C = A * B using CUDA
    auto start = std::chrono::high_resolution_clock::now();
    gemm_cuda(A, B, C_cuda, M, N, K);
    auto end = std::chrono::high_resolution_clock::now();
    auto t_cuda = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()/1000.0;

    // Compute C = A * B using CPU
    start = std::chrono::high_resolution_clock::now();
    gemm_cpu(A, B, C_cpu, M, N, K);
    end = std::chrono::high_resolution_clock::now();
    auto t_cpu = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()/1000.0;

    bool correctness = true;
    for (int i = 0; i < M * N; i++) {
        if (fabs(C_cpu[i] - C_cuda[i]) > 1e-3) {
            correctness = false;
            break;
        }
    }

    std::cout << "Time taken by CUDA:     " << t_cuda << " ms" << std::endl;
    std::cout << "Time taken by CPU:      " << t_cpu << " ms" << std::endl;
    std::cout << "Correctness:            " << correctness << std::endl;
    if (correctness) {
        std::cout << "CUDA result is correct! " << C_cuda[0] << " == " << C_cpu[0] << std::endl;
    } else {
        std::cout << "CUDA result is incorrect!" << C_cuda[0] << " != " << C_cpu[0] << std::endl;
    }
    std::cout << "Speedup:                " << t_cpu * 1.0 / t_cuda << std::endl;

}