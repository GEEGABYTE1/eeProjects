#include "utils.h"

double initWeight() {
    return ((double)rand() / RAND_MAX) * 2.0 - 1.0;


}

void freeDataset(double **inputs, double **outputs, int numSamples) {
    for (int i = 0; i < numSamples; i++) {
        free(inputs[i]);
        free(outputs[i]);
    }
    free(inputs);
    free(outputs)
}