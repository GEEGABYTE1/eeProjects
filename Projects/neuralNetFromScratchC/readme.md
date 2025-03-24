# Neural Network from Scratch in C 

Implementation of multi-layer regression neural network from scratch in C. It supports different activation functions: RELU, Sigmoid, TanH.


1. **Forward Pass:**
   - Neurons compute a weighted sum of inputs, add a bias, and apply an activation function:
     \[
     z = W \cdot x + b
     \]
     \[
     a = \sigma(z)
     \]
   - Activation Functions:
     - Sigmoid: \( \sigma(x) = \frac{1}{1 + e^{-x}} \)
     - ReLU: \( f(x) = \max(0, x) \)
     - Tanh: \( f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} \)

2. **Backward Pass (Backpropagation):**
   - Calculate the error at the output layer:
     \[
     \delta = (y - \hat{y}) \cdot \sigma'(z)
     \]
   - Propagate the error backward to adjust weights:
     \[
     W = W + \eta \cdot \delta \cdot a^T
     \]
     where \( \eta \) is the learning rate.



## Running the Model

Build the project:
```bash
gcc -o ann.c -lm
```

Run the Neural Network:
```bash 
./ann
```


## Dataset Format
The dataset is written within implementation:
```C
    double training_inputs[numTrainingSets][numInputs] = {{0.0f,0.0f},
                                                          {1.0f,0.0f},
                                                          {0.0f,1.0f},
                                                          {1.0f,1.0f}};
    double training_outputs[numTrainingSets][numOutputs] = {{0.0f},
                                                            {1.0f},
                                                            {1.0f},
                                                            {0.0f}};
```

Resulting Example (XOR dataset):
```
0 0 0
0 1 1
1 0 1
1 1 0
```

