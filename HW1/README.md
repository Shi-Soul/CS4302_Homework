# Homework 1

Weiji Xie, 2024.10.05

We successfully achieve more than 3.5x speedup with 4 threads in all problems and more than 10x speedup in p3, showing the effectiveness of OpenMP parallelization.


Code is tuned on x86_64 architecture, gcc version 9.4.0. By default it uses 4 cores (written into the code).

Correctness checking and time measurement is included for each problem.


Usage:

```shell
make 

./p1
 
./p2 100000000 50 1 2

./p3 2000 5000 21

make clean
```

Output example:

```Plain
$ ./p1
Done!
Serial time: 0.559947 seconds
Parallel time: 0.151683 seconds

$ ./p2 100000000 50 1 2
Serial Estimate of pi : 3.14172 
Parallel Estimate of pi : 3.14158 
Serial time: 1.449111 seconds
Parallel time: 0.380943 seconds

$ ./p3 2000 5000 21
Serial Convolution Time: 4.283606 seconds
Parallel Convolution Time: 0.364198 seconds
```
