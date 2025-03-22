/* neuralnet.h - Header file for neural network */
#ifndef NEURALNET_H
#define NEURALNET_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// Activation functions
double sigmoid(double x);
double dSigmoid(double x);
double relu(double x);
double dRelu(double x);
double tanh_activation(double x);
double dTanh(double x);

typedef enum {
    SIGMOID,
    RELU,
    TANH
} ActivationFunction;

typedef enum {
    SGD,
    MOMENTUM,
    RMSPROP,
    ADAM
} Optimizer;

// Neural Network Structure
typedef struct {
    int numInputs;
    int numHiddenLayers;
    int *hiddenNodes;  // Array for multiple hidden layers
    int numOutputs;
    double **hiddenWeights;
    double **hiddenBiases;
    double *outputWeights;
    double outputBias;
    ActivationFunction activation;
    double learningRate;
} NeuralNetwork;

// Neural Network Functions
NeuralNetwork* createNetwork(int numInputs, int numHiddenLayers, int *hiddenNodes, int numOutputs, double learningRate, ActivationFunction activation);
void train(NeuralNetwork *nn, double **inputs, double **outputs, int numSamples, int epochs, Optimizer optimizer);
double* predict(NeuralNetwork *nn, double *input);
void freeNetwork(NeuralNetwork *nn);
void saveModel(NeuralNetwork *nn, const char *filename);
NeuralNetwork* loadModel(const char *filename);

#endif // NEURALNET_H
