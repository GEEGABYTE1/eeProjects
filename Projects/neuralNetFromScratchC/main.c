/* Entry point for neural net exe*/

#include "neuralnet.h"
#include "dataset.h"
#include "utils.h"


int main() {

    //params 
    int numInputs = 2;
    int numOutputs = 1;
    int hiddenLayers = 1;
    int hiddenNodes[] = {2};
    double learningRate = 0.1;
    ActivationFunction activation = SIGMOID;
    Optimizer optimizer = ADAM; // change from neuralnet.c to neuralnet.h
    int epochs = 1000;

    double **inputs, **outputs;
    int numSamples;
    if (loadDataset("test.txt", &inputs, &outputs, &numSamples, numInputs, numOutputs) != 0) {
        printf("Failed to load dataset.\n");
        return 1;
    }

    NeuralNetwork *nn = createNetwork(numInputs, hiddenLayers, hiddenNodes, numOutputs, learningRate, activation);

    train(nn, inputs, outputs, numSamples, epochs, optimizer);

    printf("\nTesting Results:\n");
    for (int i = 0; i < numSamples; i++) {
        double *prediction = predict(nn, inputs[i]);
        printf("Input: %.1f %.1f => Prediction: %.2f (Expected: %.1f)\n", 
        inputs[i][0], inputs[i][1], prediction[0], outputs[i][0]
        
        );

    }

    freeDataset(inputs, outputs, numSamples);
    freeNetwork(nn);



}

