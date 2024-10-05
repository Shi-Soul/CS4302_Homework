/*
A Parallel version of Floyd's algorithm for all pairs shortest path problem, using OpenMP
*/

#include <iostream>
#include <vector>
#include <omp.h>

using namespace std;

void SerialFloyd(vector<vector<int>> &graph, int n) {
    for (int k = 0; k < n; k++) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                graph[i][j] = min(graph[i][j], graph[i][k] + graph[k][j]);
            }
        }
    }
}

void ParallelFloyd(vector<vector<int>> &graph, int n) {
    for (int k = 0; k < n; k++) {
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                graph[i][j] = min(graph[i][j], graph[i][k] + graph[k][j]);
            }
        }
    }
    #pragma omp barrier
}

int main() {
    const int N = 1000;

    vector<vector<int>> graph(N, vector<int>(N));
    // Randomly generate a graph
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            graph[i][j] = (i != j) ? (rand() % 100) : 0;
        }
    }

    // Copy the graph to a new graph for correctness check
    vector<vector<int>> graph_serial = graph;
    vector<vector<int>> graph_parallel = graph;

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
    return 0;
}

