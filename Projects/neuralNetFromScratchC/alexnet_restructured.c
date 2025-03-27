#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

//restructuring alexnet to be >1.01x more efficient (hopefully)
// updated, we reached 3x speed !!!!!!!

#define N 227 // Input size (height/width) --> more n, more pixels
#define K 11 // kernel size --> larger k, quadratic cost
#define C 3  // input channels --> more c, more memory bandwidth    
#define F 96 // output filters --> more feature maps if f inc
#define BLOCK_SIZE 32 // cache blocking size
#define GROUPS 3 // number of groups for grouped convolution


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


void depthwise_conv2d(float input[N][N][C/GROUPS], float output[N-K+1][N-K+1][C/GROUPS], 
                      float depthwise_weights[K][K][C/GROUPS]) {
    for (int c = 0; c < C/GROUPS; c++) {
        for (int i = 0; i < N - K + 1; i++) {
            for (int j = 0; j < N - K + 1; j++) {
                float sum = 0.0f;
                for (int ki = 0; ki < K; ki++) {
                    for (int kj = 0; kj < K; kj++) {
                        sum += input[i + ki][j + kj][c] * depthwise_weights[ki][kj][c];
                    }
                }
                output[i][j][c] = sum;
            }
        }
    }
}


void pointwise_conv2d(float input[N-K+1][N-K+1][C/GROUPS], float output[N-K+1][N-K+1][F/GROUPS], 
                      float pointwise_weights[C/GROUPS][F/GROUPS], float bias[F/GROUPS]) {
    for (int f = 0; f < F/GROUPS; f++) {
        for (int i = 0; i < N-K+1; i++) {
            for (int j = 0; j < N-K+1; j++) {
                float sum = 0.0f;
                for (int c = 0; c < C/GROUPS; c++) {
                    sum += input[i][j][c] * pointwise_weights[c][f];
                }
                output[i][j][f] = sum + bias[f];
            }
        }
    }
}


void grouped_conv2d(float input[N][N][C], float output[N-K+1][N-K+1][F], 
                    float weights[K][K][C/GROUPS][F/GROUPS], float bias[F]) {
    for (int g = 0; g < GROUPS; g++) {
        int channel_start = g * (C/GROUPS);
        int channel_end = (g+1) * (C/GROUPS);
        int filter_start = g * (F/GROUPS);
        int filter_end = (g+1) * (F/GROUPS);
        
        for (int f = filter_start; f < filter_end; f++) {
            for (int i = 0; i < N - K + 1; i++) {
                for (int j = 0; j < N - K + 1; j++) {
                    float sum = 0.0f;
                    for (int c = channel_start; c < channel_end; c++) {
                        for (int ki = 0; ki < K; ki++) {
                            for (int kj = 0; kj < K; kj++) {
                                sum += input[i + ki][j + kj][c] * 
                                       weights[ki][kj][c - channel_start][f - filter_start];
                            }
                        }
                    }
                    output[i][j][f] = sum + bias[f];
                }
            }
        }
    }
}


void relu_activation(float* data, int size) { //fused relu
    for (int i = 0; i < size; i++) {
        data[i] = fmaxf(0.0f, data[i]);
    }
}

int main() {
    static float input[N][N][C];
    static float output_baseline[N-K+1][N-K+1][F];
    static float output_improved[N-K+1][N-K+1][F];
    
    // initializing inputs
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int c = 0; c < C; c++) {
                input[i][j][c] = rand() / (float)RAND_MAX;
            }
        }
    }


    static float weights[K][K][C][F];
    static float bias[F];
    
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

    // improved architecture
    static float depthwise_weights[K][K][C/GROUPS];
    static float pointwise_weights[C/GROUPS][F/GROUPS];
    static float improved_bias[F/GROUPS];
    static float grouped_output[N-K+1][N-K+1][F];
    static float grouped_weights[K][K][C/GROUPS][F/GROUPS];
    

    for (int k1 = 0; k1 < K; k1++) {
        for (int k2 = 0; k2 < K; k2++) {
            for (int c = 0; c < C/GROUPS; c++) {
                depthwise_weights[k1][k2][c] = rand() / (float)RAND_MAX;
                for (int f = 0; f < F/GROUPS; f++) {
                    grouped_weights[k1][k2][c][f] = rand() / (float)RAND_MAX;
                }
            }
        }
    }
    
    for (int c = 0; c < C/GROUPS; c++) {
        for (int f = 0; f < F/GROUPS; f++) {
            pointwise_weights[c][f] = rand() / (float)RAND_MAX;
        }
    }
    
    for (int f = 0; f < F/GROUPS; f++) {
        improved_bias[f] = rand() / (float)RAND_MAX;
    }

    clock_t start_improved = clock();
    

    grouped_conv2d(input, grouped_output, grouped_weights, bias);
    relu_activation(&grouped_output[0][0][0], (N-K+1)*(N-K+1)*F);

    clock_t end_improved = clock();
    double improved_time = (double)(end_improved - start_improved) / CLOCKS_PER_SEC;

    printf("Baseline Convolution Time: %.4f seconds\n", baseline_time);
    printf("Improved Architecture Time: %.4f seconds\n", improved_time);
    printf("Speedup: %.2fx\n", baseline_time / improved_time);

    return 0;
}