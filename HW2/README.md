
# Homework 1

Weiji Xie, 2024.11.03

## Environment

```
GPU: NVIDIA GeForce RTX 2080 Ti
nvcc V12.2.140
```

## Prob 1: GEMM

```shell
make gemm

./gemm 1024 1024 1024
```

Output Example:
```
$ ./gemm 1024 1024 1024
Time taken by CUDA:     307.998 ms
Time taken by CPU:      3129.59 ms
Correctness:            1
CUDA result is correct! 411 == 411
Speedup:                10.1611



$ ./gemm 2048 2048 2048
Time taken by CUDA:     325.28 ms
Time taken by CPU:      26664.7 ms
Correctness:            1
CUDA result is correct! 624 == 624
Speedup:                81.9746

```

## Prob 2: Array Sum

To run the program:
```shell
make sum

./sum 100000000
```

Output Example:
```
num_blocks: 390625
Time taken by CUDA:     363 ms
Time taken by CPU:      104 ms
Correctness: 1
CUDA result is correct! 22338 == 22338
Speedup: 0.288446
```