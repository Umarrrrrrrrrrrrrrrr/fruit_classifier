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
        Z = inputs @ self.weights[1:].T + self.weight[0]
        return Z
    
    # Define the Heavyside step function
    def Heaviside_step_fn(self, z):
        if z>=0:
            return 1
        else:
            return 0
        
    # Define the prediction
    def predict(self, inputs):
        Z = self.linear(inputs)
        try:
            pred = []
            for z in Z:
                pred.append(self.Heaviside_step_fn(z))
        except:
            return self.Heaviside_step_fn(Z)
        return pred
    
    # Define the loss function
    def loss(self, prediction, target):
        loss = (prediction - target)
        return loss
    
    # Define traning
    def train(self, inputs, target):
        prediction = self.predict(inputs)
        error = self.loss(prediction, target)
        self.weights[1:] += self.learning_rate * error * inputs
        self.weights[0] += self.learning_rate * error

    # fit the model
    def fit(self, X, y, num_epochs):
        for epoch in range(num_epochs):
            for inputs, target in zip(X, y):
                self.train(inputs, target)