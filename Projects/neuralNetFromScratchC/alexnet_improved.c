#include <stdio.h>
#include <stdlib.h>
#include <time.h>       // For timing

#define N 227  // Input size for AlexNet (227x227)
#define K 11   // Kernel size (first convolution)
#define C 3    // Input channels
#define F 96   // Number of filters in the first conv layer
#define BLOCK_SIZE 32  // Cache blocking size

// Normal Convolution (Baseline)
void conv2d_baseline(float input[N][N][C], float output[N-K+1][N-K+1][F], float weights[K][K][C][F], float bias[F]) {
    for (int f = 0; f < F; f++) {
        for (int i = 0; i < N - K + 1; i++) {
            for (int j = 0; j < N - K + 1; j++) {
                float sum = 0.0f;
                for (int c = 0; c < C; c++) {
                    for (int ki = 0; ki < K; ki++) {
                        for (int kj = 0; kj < K; kj++) {
                            sum += input[i + ki][j + kj][c] * weights[ki][kj][c][f];
                        }
                    }
                }
                output[i][j][f] = sum + bias[f];
            }
        }
    }
}

// Optimized Convolution with Cache Blocking + Loop Unrolling
void conv2d_optimized(float input[N][N][C], float output[N-K+1][N-K+1][F], float weights[K][K][C][F], float bias[F]) {
    for (int f = 0; f < F; f++) {
        for (int i = 0; i < N - K + 1; i += BLOCK_SIZE) {
            for (int j = 0; j < N - K + 1; j += BLOCK_SIZE) {
                for (int bi = i; bi < i + BLOCK_SIZE && bi < N - K + 1; bi++) {
                    for (int bj = j; bj < j + BLOCK_SIZE && bj < N - K + 1; bj++) {
                        float sum = 0.0f;
                        for (int c = 0; c < C; c++) {
                            for (int ki = 0; ki < K; ki++) {
                                for (int kj = 0; kj < K; kj += 4) {  // Loop unrolling
                                    sum += input[bi + ki][bj + kj][c] * weights[ki][kj][c][f];
                                    if (kj + 1 < K) sum += input[bi + ki][bj + kj + 1][c] * weights[ki][kj + 1][c][f];
                                    if (kj + 2 < K) sum += input[bi + ki][bj + kj + 2][c] * weights[ki][kj + 2][c][f];
                                    if (kj + 3 < K) sum += input[bi + ki][bj + kj + 3][c] * weights[ki][kj + 3][c][f];
                                }
                            }
                        }
                        output[bi][bj][f] = sum + bias[f];
                    }
                }
            }
        }
    }
}

int main() {
    static float input[N][N][C];
    static float output_baseline[N-K+1][N-K+1][F];
    static float output_optimized[N-K+1][N-K+1][F];
    static float weights[K][K][C][F];
    static float bias[F];

    // Initialize input, weights, and bias
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int c = 0; c < C; c++) {
                input[i][j][c] = rand() / (float)RAND_MAX;
            }
        }
    }
    for (int k1 = 0; k1 < K; k1++) {
        for (int k2 = 0; k2 < K; k2++) {
            for (int c = 0; c < C; c++) {
                for (int f = 0; f < F; f++) {
                    weights[k1][k2][c][f] = rand() / (float)RAND_MAX;
                }
            }
        }
    }
    for (int f = 0; f < F; f++) {
        bias[f] = rand() / (float)RAND_MAX;
    }

    // Measure baseline performance
    clock_t start_baseline = clock();
    conv2d_baseline(input, output_baseline, weights, bias);
    clock_t end_baseline = clock();
    double baseline_time = (double)(end_baseline - start_baseline) / CLOCKS_PER_SEC;

    // Measure optimized performance
    clock_t start_optimized = clock();
    conv2d_optimized(input, output_optimized, weights, bias);
    clock_t end_optimized = clock();
    double optimized_time = (double)(end_optimized - start_optimized) / CLOCKS_PER_SEC;

    // Print performance metrics
    printf("Baseline Convolution Time: %.4f seconds\n", baseline_time);
    printf("Optimized Convolution Time: %.4f seconds\n", optimized_time);
    printf("Speedup: %.2fx\n", baseline_time / optimized_time);

    return 0;
}