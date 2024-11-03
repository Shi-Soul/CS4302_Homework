/*
Array Sum in CUDA
*/

#include <iostream>
#include <cuda_runtime.h>
#include <cassert>
#include <chrono>

const int BLOCK_SIZE = 1024;


__global__ void array_sum_v0(float *arr, int N, float *result) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        // result[0] += arr[tid];
        atomicAdd(result, arr[tid]);
    }
}

__global__ void array_sum_v1(float *arr, int N, float *result) {
    __shared__ float partialSum[BLOCK_SIZE];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    partialSum[threadIdx.x] = (tid < N) ? arr[tid] : 0;
    __syncthreads();
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            partialSum[threadIdx.x] += partialSum[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        result[blockIdx.x] = partialSum[0];
    }
}

void array_sum_cuda(float *arr, int N, float *result) {
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;


    float *d_arr, *d_result, *h_buffer;
    h_buffer = new float[num_blocks];
    cudaMalloc((void**)&d_arr, N * sizeof(float));
    cudaMalloc((void**)&d_result, num_blocks * sizeof(float));
    cudaMemcpy(d_arr, arr, N * sizeof(float), cudaMemcpyHostToDevice);


    array_sum_v1<<<num_blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_arr, N, d_result);


    cudaMemcpy(h_buffer, d_result, num_blocks * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 1; i < num_blocks; i++) {
        h_buffer[0] += h_buffer[i];
    }

    cudaMemcpy(result, h_buffer, sizeof(float), cudaMemcpyHostToHost);
    cudaFree(d_arr);
    cudaFree(d_result);
}


// Claude's code
template<unsigned int blockSize>
__global__ void sumReduceKernel(const float* __restrict__ input, float* __restrict__ output, unsigned int n) {
    extern __shared__ float sdata[];
    
    // Each thread loads one element from global to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * 2) + tid;
    
    // Load and perform first add of global data into shared memory
    float sum = 0;
    if (i < n)
        sum = input[i];
    if (i + blockSize < n)
        sum += input[i + blockSize];
    sdata[tid] = sum;
    __syncthreads();

    // Perform reduction in shared memory
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
    
    // Unroll last 6 iterations
    if (tid < 32) {
        volatile float* smem = sdata;
        if (blockSize >= 64) smem[tid] += smem[tid + 32];
        if (blockSize >= 32) smem[tid] += smem[tid + 16];
        if (blockSize >= 16) smem[tid] += smem[tid + 8];
        if (blockSize >= 8) smem[tid] += smem[tid + 4];
        if (blockSize >= 4) smem[tid] += smem[tid + 2];
        if (blockSize >= 2) smem[tid] += smem[tid + 1];
    }
    
    // Write result for this block to global memory
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// Host function to perform array summation
float sumArray(const float* d_input, int n) {
    const int threadsPerBlock = 256;
    int numBlocks = (n + threadsPerBlock * 2 - 1) / (threadsPerBlock * 2);
    
    // Allocate memory for partial sums
    float* d_partialSums;
    cudaMalloc(&d_partialSums, numBlocks * sizeof(float));
    
    // First reduction step
    sumReduceKernel<threadsPerBlock><<<numBlocks, threadsPerBlock, threadsPerBlock * sizeof(float)>>>
        (d_input, d_partialSums, n);
    
    // If there are multiple blocks, reduce partial sums on CPU
    float finalSum = 0;
    if (numBlocks > 1) {
        float* h_partialSums = new float[numBlocks];
        cudaMemcpy(h_partialSums, d_partialSums, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);
        
        for (int i = 0; i < numBlocks; i++) {
            finalSum += h_partialSums[i];
        }
        delete[] h_partialSums;
    } else {
        cudaMemcpy(&finalSum, d_partialSums, sizeof(float), cudaMemcpyDeviceToHost);
    }
    
    cudaFree(d_partialSums);
    return finalSum;
}

///
void Claude_main(float* h_input, int N, float*result) {
    float *d_input;
    cudaMalloc((void**)&d_input, N * sizeof(float));
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    result[0] = sumArray(d_input, N);
    cudaFree(d_input);
}

void array_sum_cpu(float *arr, int N, float *result) {
    for (int i = 0; i < N; i++) {
        result[0] += arr[i];
    }
}

int main(int argc, char **argv) {

    int N;
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <N>" << std::endl;
        return 1;
    }
    N = atoi(argv[1]);

    float *arr = new float[N];
    for (int i = 0; i < N; i++) {
        arr[i] = rand() % 10;
    }

    float *result_cuda = new float[1];
    result_cuda[0] = 0;

    float *result_cpu = new float[1];
    result_cpu[0] = 0;

    // array_sum_cpu(arr, N, result_cpu);
    // std::cout << "CPU result: " << result_cpu[0] << std::endl;


    auto start = std::chrono::high_resolution_clock::now();
    array_sum_cuda(arr, N, result_cuda);
    // Claude_main(arr, N, result_cuda);
    auto end = std::chrono::high_resolution_clock::now();
    auto t_cuda = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Time taken by CUDA: \t" << t_cuda << " ms" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    array_sum_cpu(arr, N, result_cpu);
    end = std::chrono::high_resolution_clock::now();
    auto t_cpu = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Time taken by CPU: \t" << t_cpu << " ms" << std::endl;

    bool correctness = fabs(result_cuda[0]- result_cpu[0])/result_cpu[0] < 1e-3;
    std::cout << "Correctness: " << correctness << std::endl;
    if (!correctness) {
        std::cerr << "CUDA result is incorrect! " << result_cuda[0] << " != " << result_cpu[0] << std::endl;
        return 1;
    }
    float speedup = 1.0*t_cpu / t_cuda;
    std::cout << "Speedup: " << speedup << std::endl;
}