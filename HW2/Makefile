.PHONY: all gemm sum clean

all: gemm sum

gemm: gemm.cu
	nvcc -O3 -o gemm gemm.cu

sum: sum.cu
	nvcc -o sum sum.cu

clean:
	rm -f gemm sum