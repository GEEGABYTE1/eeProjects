//improved neural net with dynamic functionality

#include "neuralnet.h"

// activation functions and its derivatives 
double sigmoid(double x) {return 1 / 1 + exp(-x);}
double dSigmoid(double x) {return x * (1 - x);}
double relu(double x) {return x > 0 ? x : 0;}
double dRelu(double x) {return x > 0 ? 1 : 0;}
double tanh_activation(double x) {return tanh(x);}
double dTanh(double x) {return 1 - x * x;}

//optimizer enum
typedef enum {
    SGD,
    MOMENTUM,
    RMSPROP,
    ADAM 
} Optimizer;

//initialize neural network 

NeuralNetwork* createNetwork(int numInputs, int numHiddenLayers, int *hiddenNodes, int numOutputs, double learningRate, ActivationFunction activation) {
    NeuralNetwork *nn = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    nn->numInputs = numInputs;
    nn->numHiddenLayers = numHiddenLayers;
    nn->hiddenNodes = hiddenNodes;
    nn->numOutputs = numOutputs;
    nn->learningRate = learningRate;
    nn->activation = activation;

    srand(time(NULL));

    //allocating weight and biases 
    nn->hiddenWeights = (double **)malloc(numHiddenLayers * sizeof(double *));
    nn->hiddenBiases = (double **)malloc(numHiddenLayers * sizeof(double *));
    for (int l =0; l < numHiddenLayers; l++) {
        int inputSize = (l == 0) ? numInputs : hiddenNodes[l-1];
        nn->hiddenWeights[l] = (double *)malloc(inputSize * hiddenNodes[l] * sizeof(double));   
        nn->hiddenBiases[l] = (double *)malloc(hiddenNodes[l] * sizeof(double));
        for (int j=0; j < hiddenNodes[l]; j++) {
            nn->hiddenBiases[l][j] = (double)rand() / RAND_MAX;
            for (int i=0; i < inputSize; i++) {
                nn->hiddenWeights[l][j*inputSize + i] = (double)rand() / RAND_MAX;
            }
        }
    }

    //outputting weights and biases
    nn->outputWeights = (double *)malloc(hiddenNodes[numHiddenLayers-1]*sizeof(double));
    nn->outputBias = (double)rand() / RAND_MAX;
    for (int i =0; i < hiddenNodes[numHiddenLayers-1]; i++) {
        nn->outputWeights[i] = (double)rand() / RAND_MAX;
    }

    return nn;
}



//training the thing
void train(NeuralNetwork *nn, double **inputs, double **outputs, int numSamples, int epochs, Optimizer optimizer) {
    double momentum = 0.9, beta1=0.9, beta2=0.999, epsilon=1e-8;
    double *velocity = (double *)calloc(nn->hiddenNodes[nn->numHiddenLayers-1], sizeof(double));
    double *squaredGrad = (double *)calloc(nn->hiddenNodes[nn->numHiddenLayers-1], sizeof(double));

    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int sample =0; sample < numSamples; sample++) {
            double *output = predict(nn, inputs[sample]);
            double error = outputs[sample][0] - output[0];
            double dOutput = error * dSigmoid(output[0]);

            //updating output weights given optimizer
            for (int i =0; i < nn->hiddenNodes[nn->numHiddenLayers-1]; i++) {
                double grad = nn->learningRate * dOutput * inputs[sample][i];
                if (optimizer == MOMENTUM) {
                    velocity[i] = momentum * velocity[i] + grad;
                    nn->outputWeights[i] += velocity[i];
                } else if (optimizer == RMSPROP) {
                    squaredGrad[i] = beta2 * squaredGrad[i] + (1 - beta2) * grad * grad;
                    nn->outputWeights[i] += grad / (sqrt(squaredGrad[i]) + epsilon);
                } else if (optimizer == ADAM) {
                    velocity[i] = beta1 * velocity[i] + (1 - beta1) * grad;
                    squaredGrad[i] = beta2 * squaredGrad[i] + (1-beta2) * grad * grad;
                    nn->outputWeights[i] += (velocity[i] / sqrt(squaredGrad[i] + epsilon));
                } else  { // SGD 
                    nn->outputWeights[i] += grad;

                }
                
            }

            nn->outputBias += nn->learningRate * dOutput;
            free(output);
        }
    }
    free(velocity);
    free(squaredGrad);
}

void freeNetwork(NeuralNetwork *nn) {
    for (int l =0; l< nn->numHiddenLayers; l++) {
        free(nn->hiddenWeights[l]);
        free(nn->hiddenBiases[l]); 
    }
    free(nn->hiddenWeights);
    free(nn->hiddenBiases);
    free(nn->outputWeights);
    free(nn);
}   