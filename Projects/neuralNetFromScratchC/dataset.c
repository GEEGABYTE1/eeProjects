#include "dataset.h"

int loadDataset(const char *filename, double ***inputs, double ***outputs, int *numSamples, int inputSize, int outputSize) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Failed to opne dataset file");
        return -1;
    } 

    fscanf(file, "%d", numSamples);
    *inputs = (double **)malloc(*numSamples * sizeof(double *));
    *outputs = (double **)malloc(*numSamples * sizeof(double *)); 

    for (int i = 0; i < *numSamples; i++) {
        (*inputs)[i] = (double *)malloc(inputSize * sizeof(double));
        (*outputs)[i] = (double *)malloc(outputSize * sizeof(double));
        for (int j = 0; j < inputSize; j++) {
            fscanf(file, "%lf", &(*inputs)[i][j]);
        }
        for (int k =0; k < outputSize; k++) {
            fscanf(file, "%lf", &(*outputs)[i][k]);
        }
    }

    fclose(file);
    return 0;
}


void shuffleDataset(double **inputs, double **outputs, int numSamples) {
    srand(time(NULL));
    for (int i = numSamples - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        double *tempInput = inputs[i];
        double *tempOutput = outputs[i];
        inputs[i] = inputs[j];
        outputs[i] = outputs[j];
        inputs[j] = tempInput;
        outputs[j] = tempOutput;
    }
}