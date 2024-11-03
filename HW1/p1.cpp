/*
A Parallel version of Floyd's algorithm for all pairs shortest path problem, using OpenMP
*/

#include <iostream>
#include <omp.h>

using namespace std;

const int NUM_THREADS = 4;

void SerialFloyd(int **graph, int n) {
    for (int k = 0; k < n; k++) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                graph[i][j] = min(graph[i][j], graph[i][k] + graph[k][j]);
            }
        }
    }
}

void ParallelFloyd(int **graph, int n) {
    for (int k = 0; k < n; k++) {
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                graph[i][j] = min(graph[i][j], graph[i][k] + graph[k][j]);
            }
        }
    }
}

int main() {
    omp_set_num_threads(NUM_THREADS); 
    const int N = 1000;
    int **graph = new int*[N];
    int **graph_serial = new int*[N];
    int **graph_parallel = new int*[N];

    for (int i = 0; i < N; i++) {
        graph[i] = new int[N];
        graph_serial[i] = new int[N];
        graph_parallel[i] = new int[N];
        for (int j = 0; j < N; j++) {
            graph[i][j] = (i != j) ? (rand() % 100) : 0;
            graph_serial[i][j] = graph[i][j];
            graph_parallel[i][j] = graph[i][j];
        }
    }

    double t1 = omp_get_wtime();
    SerialFloyd(graph_serial, N);
    double t2 = omp_get_wtime();
    ParallelFloyd(graph_parallel, N);
    double t3 = omp_get_wtime();

    // Check if the parallel result is the same as the serial result
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (graph_serial[i][j] != graph_parallel[i][j]) {
                cout << "Mismatch found at graph[" << i << "][" << j << "]: " << graph_serial[i][j] << " != " << graph_parallel[i][j] << endl;
            }
        }
    }

    cout << "Done!" << endl;
    cout << "Serial time: " << t2 - t1 << " seconds" << endl;
    cout << "Parallel time: " << t3 - t2 << " seconds" << endl;

    // Cleanup dynamically allocated memory
    for (int i = 0; i < N; i++) {
        delete[] graph[i];
        delete[] graph_serial[i];
        delete[] graph_parallel[i];
    }
    delete[] graph;
    delete[] graph_serial;
    delete[] graph_parallel;

    return 0;
}

