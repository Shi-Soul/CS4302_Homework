/*
A 2D convolution, using OpenMP
*/

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define NUM_THREADS 4
#define BLOCK_SIZE 16

void SerialConvolution(int M, int N, int K, double **input, double **kernel, double **output){
    int half_k = K / 2;
    for (int i = 0; i < M-K+1; i++) {
        for (int j = 0; j < N-K+1; j++) {
            output[i][j] = 0;

            for (int x = 0; x < K; x++) {
                for (int y = 0; y < K; y++) {
                    output[i][j] += input[i + x][j + y] * kernel[x][y];
                }
            }
        }
    }
}


void ParallelConvolution(int M, int N, int K, double **input, double **kernel, double **output){
    int half_k = K / 2;
    
    // Thread and cache blocking
    int blockSize = BLOCK_SIZE;  // Block size can be tuned for the architecture

    #pragma omp parallel for collapse(2) num_threads(NUM_THREADS) schedule(static)
    for (int i = 0; i < M-K+1; i += blockSize) {
        for (int j = 0; j < N-K+1; j += blockSize) {

            // Loop over blocks
            for (int ii = i; ii < i + blockSize && ii < M-K+1; ii++) {
                for (int jj = j; jj < j + blockSize && jj < N-K+1; jj++) {
                    double sum = 0.0;
                    
                    // Perform the convolution operation
                    #pragma omp simd reduction(+:sum) aligned(input, kernel: 64) 
                    for (int x = 0; x < K; x++) {
                        for (int y = 0; y < K; y++) {
                            sum += input[ii + x][jj + y] * kernel[x][y];
                        }
                    }
                    output[ii][jj] = sum;
                }
            }
        }
    }
}


int main(int argc, char *argv[]){
    int M, N, K;
    double **input, **kernel, **output1, **output2;
    double epsilon = 1e-15;

    // read input from command line
    if (argc != 4) {
        printf("Usage: %s M N K\n", argv[0]);
        return 1;
    }
    M = atoi(argv[1]);
    N = atoi(argv[2]);
    K = atoi(argv[3]);
    // ensure K is odd
    if (K % 2 == 0) {
        printf("Error: K must be odd\n");
        return 1;
    }

    // Randomly initialize input, kernel, and output
    input = (double **)malloc(M * sizeof(double *)); // M x N
    for (int i = 0; i < M; i++) {
        input[i] = (double *)malloc(N * sizeof(double));
        for (int j = 0; j < N; j++) {
            input[i][j] = rand() % 100;
        }
    }

    kernel = (double **)malloc(K * sizeof(double *)); // K x K
    for (int i = 0; i < K; i++) {
        kernel[i] = (double *)malloc(K * sizeof(double));
        for (int j = 0; j < K; j++) {
            kernel[i][j] = rand() % 100;
        }
    }

    output1 = (double **)malloc((M-K+1) * sizeof(double *)); // (M-K+1) x (N-K+1)
    output2 = (double **)malloc((M-K+1) * sizeof(double *));
    for (int i = 0; i < M-K+1; i++) {
        output1[i] = (double *)malloc((N-K+1) * sizeof(double));
        output2[i] = (double *)malloc((N-K+1) * sizeof(double));
    }
    
    double t1 = omp_get_wtime();
    SerialConvolution(M, N, K, input, kernel, output1);
    double t2 = omp_get_wtime();
    ParallelConvolution(M, N, K, input, kernel, output2);
    double t3 = omp_get_wtime();

    // Correctness check
    for (int i = 0; i < M-K+1; i++) {
        for (int j = 0; j < N-K+1; j++) {
            if (fabs(output1[i][j] - output2[i][j]) > epsilon) {
                printf("Error: output1[%d][%d] = %f, output2[%d][%d] = %f\n", i, j, output1[i][j], i, j, output2[i][j]);
            }
        }
    }

    printf("Serial Convolution Time: %f seconds\n", t2 - t1);
    printf("Parallel Convolution Time: %f seconds\n", t3 - t2);
    return 0;
}