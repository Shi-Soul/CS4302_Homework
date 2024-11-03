/*
A parallel Monte Carlo simulation to estimate the value of pi using OpenMP
*/

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define NUM_THREADS 4




void SerialMonteCarlo(int samples, unsigned short xi[3])
{

    int count; // points inside the circle
    int i;
    double x, y;

    count = 0;
    {
        for (i = 0; i < samples; i++)
        {
            x = erand48(xi);
            y = erand48(xi);
            if (x * x + y * y <= 1.0)
                count++;
        }
    }
    printf("Serial Estimate of pi : %7.5f \n", 4.0 * count / samples);
}
void ParallelMonteCarlo(int samples, unsigned short xi[3])
{
    int tot_count; // points inside the circle
    int i;
    double x, y;

    tot_count = 0;
    #pragma omp parallel reduction(+:tot_count) private(i,x,y) num_threads(NUM_THREADS)
    {
        int tid = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        unsigned short int priv_xi[3] = {xi[0] + tid, xi[1] + tid, xi[2] + tid};
        for (i = tid; i < samples; i += num_threads)
        {
            x = erand48(priv_xi);
            y = erand48(priv_xi);
            if (x * x + y * y <= 1.0)
                tot_count++;
        }
    }

    printf("Parallel Estimate of pi : %7.5f \n", 4.0 * tot_count / samples);
}


int main(int argc, char *argv[])
{
    if (argc != 5)
    {
        printf("Usage: %s <samples> <xi1> <xi2> <xi3>\n", argv[0]);
        return 1;
    }

    int samples = atoi(argv[1]);
    unsigned short xi[3] = {atoi(argv[2]), atoi(argv[3]), atoi(argv[4])};

    double t1 = omp_get_wtime();
    SerialMonteCarlo(samples, xi);
    double t2 = omp_get_wtime();
    ParallelMonteCarlo(samples, xi);
    double t3 = omp_get_wtime();

    printf("Serial time: %f seconds\n", t2 - t1);
    printf("Parallel time: %f seconds\n", t3 - t2);
    return 0;
}