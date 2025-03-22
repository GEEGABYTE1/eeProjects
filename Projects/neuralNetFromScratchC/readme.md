# Neural Network from Scratch in C 

Implementation of multi-layer regression neural network from scratch in C. It supports different activation functions and optimizers: SGD, Momentum, RMSprop, Adam.


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

3. **Optimizers:**
   - Momentum: Accelerates SGD by adding a fraction of the previous update.
   - RMSprop: Adapts learning rate by scaling gradients based on recent magnitudes.
   - Adam: Combines momentum and RMSprop for more efficient convergence.


## Running the Model

Build the project:
```bash
make
```

Run the Neural Network:
```bash 
./neuralnet
```

Plotting the results
```
run all code cells in viz.ipynb
```

Cleaning up
```bash
make clean
```

## Dataset Format
The dataset should be a *.txt* file with the following format:
```
<num_samples>
<input_1> <input_2> ... <input_n> <output_1> <output_2> ... <output_m>
```

Example (XOR dataset):
```
4
0 0 0
0 1 1
1 0 1
1 1 0
```

