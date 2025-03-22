/* neuralnet.c - Core neural network functions */
#include "neuralnet.h"

// Activation Functions and Derivatives
double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
double dSigmoid(double x) { return x * (1.0 - x); }
double relu(double x) { return x > 0 ? x : 0; }
double dRelu(double x) { return x > 0 ? 1 : 0; }
double tanh_activation(double x) { return tanh(x); }
double dTanh(double x) { return 1 - pow(tanh(x), 2); }



// Initialize Neural Network
NeuralNetwork* createNetwork(int numInputs, int numHiddenLayers, int *hiddenNodes, int numOutputs, double learningRate, ActivationFunction activation) {
    NeuralNetwork *nn = (NeuralNetwork *)malloc(sizeof(NeuralNetwork));
    nn->numInputs = numInputs;
    nn->numHiddenLayers = numHiddenLayers;
    nn->hiddenNodes = hiddenNodes;
    nn->numOutputs = numOutputs;
    nn->learningRate = learningRate;
    nn->activation = activation;

    srand(time(NULL));

    // Allocate weights and biases
    nn->hiddenWeights = (double **)malloc(numHiddenLayers * sizeof(double *));
    nn->hiddenBiases = (double **)malloc(numHiddenLayers * sizeof(double *));
    for (int l = 0; l < numHiddenLayers; l++) {
        int inputSize = (l == 0) ? numInputs : hiddenNodes[l - 1];
        nn->hiddenWeights[l] = (double *)malloc(inputSize * hiddenNodes[l] * sizeof(double));
        nn->hiddenBiases[l] = (double *)malloc(hiddenNodes[l] * sizeof(double));
        for (int j = 0; j < hiddenNodes[l]; j++) {
            nn->hiddenBiases[l][j] = (double)rand() / RAND_MAX;
            for (int i = 0; i < inputSize; i++) {
                nn->hiddenWeights[l][j * inputSize + i] = (double)rand() / RAND_MAX;
            }
        }
    }
    // Output weights and biases
    nn->outputWeights = (double *)malloc(hiddenNodes[numHiddenLayers - 1] * sizeof(double));
    nn->outputBias = (double)rand() / RAND_MAX;
    for (int i = 0; i < hiddenNodes[numHiddenLayers - 1]; i++) {
        nn->outputWeights[i] = (double)rand() / RAND_MAX;
    }

    return nn;
}

// Predict function
double* predict(NeuralNetwork *nn, double *input) {
    double *currentInput = input;

    // Forward pass through hidden layers
    for (int l = 0; l < nn->numHiddenLayers; l++) {
        double *hiddenOutput = (double *)malloc(nn->hiddenNodes[l] * sizeof(double));
        int inputSize = (l == 0) ? nn->numInputs : nn->hiddenNodes[l - 1];

        for (int j = 0; j < nn->hiddenNodes[l]; j++) {
            double activation = nn->hiddenBiases[l][j];
            for (int i = 0; i < inputSize; i++) {
                activation += currentInput[i] * nn->hiddenWeights[l][j * inputSize + i];
            }

            // Apply activation function
            if (nn->activation == SIGMOID) {
                hiddenOutput[j] = sigmoid(activation);
            } else if (nn->activation == RELU) {
                hiddenOutput[j] = relu(activation);
            } else if (nn->activation == TANH) {
                hiddenOutput[j] = tanh_activation(activation);
            }
        }
        if (l > 0) free(currentInput);
        currentInput = hiddenOutput;
    }

    // Output layer
    double *output = (double *)malloc(nn->numOutputs * sizeof(double));
    double activation = nn->outputBias;
    for (int i = 0; i < nn->hiddenNodes[nn->numHiddenLayers - 1]; i++) {
        activation += currentInput[i] * nn->outputWeights[i];
    }
    output[0] = sigmoid(activation); // Output activation is sigmoid

    free(currentInput);
    return output;
}

// Free memory
void freeNetwork(NeuralNetwork *nn) {
    for (int l = 0; l < nn->numHiddenLayers; l++) {
        free(nn->hiddenWeights[l]);
        free(nn->hiddenBiases[l]);
    }
    free(nn->hiddenWeights);
    free(nn->hiddenBiases);
    free(nn->outputWeights);
    free(nn);
}
