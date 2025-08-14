# importing the necessary Libraryu
import numpy as np

# Build the perceptron model
class Perceptron:
    def __init__(self, num_inputs, learning_rate=0.01):
    #Initilize the weight and lerning rate
        self.weights = np.random.rand(num_inputs + 1)
        self.learning_rate = learning_rate

    # Define the first linear layer
    def linear(self, inputs):
        Z = inputs @ self.weights[1:].T + + self.weight[0]
        return Z
    
    # Define the Heavyside step function
    def Heaviside_step_fn(self, z):
        if z>=0:
            return 1
        else:
            return 0
        
    # Define the prediction
    