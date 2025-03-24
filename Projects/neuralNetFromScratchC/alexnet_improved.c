#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <time.h> 

#define N 227 /* input size */
#define K 11 /*kernel size (first convolution) */
#define C 3 /* input channels */
#define F 96 /* cache blocking size */
#define BLOCK_SIZE 32

void conv2d_baseline(float input[N][N][C], float output[N-K+1][N-K+1][F], float weights[K][K][C][F], float bias[F]) {
    for (int f =0; f < F; f++) {
        for (int i = 0; i < N-K+1; i++) {
            for (int j =0; j < N-K+1; j++) {
                float sum = 0.0f;
                for (int c = 0; c < C; c++) {
                    for (int ki = 0; ki < K; ki++) {
                        for (int kj = 0; kj < K; kj++) {
                            sum += input[i+ki][j+kj][c] * weights[ki][kj][c][f];
                        }
                    }
                }
                output[i][j][f] = sum + bias[f];
            }
        }
    }
}

void conv2d_improved(float input[N][N][C], float output[N-K+1][N-K+1][F], float weights[K][K][C][F], float bias[F]) {
    for (int f =0; f < F; f++) {
        for (int i = 0; i < N- K  + 1; i += BLOCK_SIZE) {
            for (int j =0; j < N-K+1; j += BLOCK_SIZE) {
                for (int bi = i; bi < i + BLOCK_SIZE && bi < N - K + 1; bi++) {
                    for (int bj = j; bj < j + BLOCK_SIZE && bj < N - K + 1; bj++) {
                        __m256 sum = _mm256_setzero_ps();
                        for (int c =0; c < C; c++) {
                            for (int ki = 0; ki < K; ki++) {
                                for (int kj = 0; kj < K; kj++) {
                                    sum = _mm256_add_ps(sum, _mm256_set1_ps(input[bi + ki][bi + kj][c]) * _mm256_set1_ps(weights[ki][kj][c][f]));
                                }
                            }
                        }
                        float result[8];
                        _mm256_storeu_ps(result, sum);
                        output[bi][bj][f] = result[0] + result[1] + result[2] + result[3] + result[4] + result[5] + result[6] + result[7] + result[8] + result[9];
 
                    }
                }
            }
        }
    }
}