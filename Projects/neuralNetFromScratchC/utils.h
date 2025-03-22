/* utils.h - utility functions */

#ifndef UTILS_H
#define UTILS_H 

#include <stdlib.h>
#include <math.h> 

// random weight initialization
double initWeight(); 

void freeDataset(double **inputs, double **outputs, int numSamples); 

#endif // UTILS_H