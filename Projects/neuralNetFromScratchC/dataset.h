/* dataset.h - Dataset handling functions */
#ifndef DATASET_H
#define DATASET_H 

#include <stdio.h>  
#include <stdlib.h>
#include <time.h>

int loadDataset(const char *filename, double ***inputs, double ***outputs, int *numSamples, int inputSize, int outputSize);
void shuffleDataset(double **inputs, double **outputs, int numSamples); 

#endif // DATASET_H


