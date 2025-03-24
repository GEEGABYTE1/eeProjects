#include <stdio.h>
#include <stdlib.h>
#include <arm_neon.h>  
#include <time.h>       

#define N 227  // input
#define K 11   // kernel
#define C 3    // Input channels
#define F 96   // filters for conv layer
#define BLOCK_SIZE 8  


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

// optimized with SIMD
void conv2d_optimized(float input[N][N][C], float output[N-K+1][N-K+1][F], float weights[K][K][C][F], float bias[F]) {
    for (int f = 0; f < F; f++) {
        for (int i = 0; i < N - K + 1; i += BLOCK_SIZE) {
            for (int j = 0; j < N - K + 1; j += BLOCK_SIZE) {
                for (int bi = i; bi < i + BLOCK_SIZE && bi < N - K + 1; bi++) {
                    for (int bj = j; bj < j + BLOCK_SIZE && bj < N - K + 1; bj++) {
                        float32x4_t sum_vec = vdupq_n_f32(0.0f);
                        for (int c = 0; c < C; c++) {
                            for (int ki = 0; ki < K; ki++) {
                                for (int kj = 0; kj < K; kj++) {
                                    float val = input[bi + ki][bj + kj][c];
                                    float w = weights[ki][kj][c][f];
                                    sum_vec = vaddq_f32(sum_vec, vmulq_n_f32(vdupq_n_f32(val), w));
                                }
                            }
                        }
                        float sum = vgetq_lane_f32(sum_vec, 0) + vgetq_lane_f32(sum_vec, 1) + vgetq_lane_f32(sum_vec, 2) + vgetq_lane_f32(sum_vec, 3);
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


    clock_t start_baseline = clock();
    conv2d_baseline(input, output_baseline, weights, bias);
    clock_t end_baseline = clock();
    double baseline_time = (double)(end_baseline - start_baseline) / CLOCKS_PER_SEC;

    // Measure optimized performance
    clock_t start_optimized = clock();
    conv2d_optimized(input, output_optimized, weights, bias);
    clock_t end_optimized = clock();
    double optimized_time = (double)(end_optimized - start_optimized) / CLOCKS_PER_SEC;

    printf("Baseline Convolution Time: %.4f seconds\n", baseline_time);
    printf("Optimized Convolution Time: %.4f seconds\n", optimized_time);
    printf("Speedup: %.2fx\n", baseline_time / optimized_time);

    return 0;
}
