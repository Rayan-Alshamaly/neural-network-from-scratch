import numpy as np
import random
from math import *

def sigmoid(x):
    return 1/(1+exp(-x))

def derivative_sigmoid(sigmoid_y):
    return sigmoid_y*(1-sigmoid_y)

class Perceptron:
    def __init__(self,n_inputs,weights=None):
        self.n = n_inputs
        if weights is not None:
            self.weights = weights
        else:
            self.weights = np.random.uniform(-1, 1, n_inputs + 1)
        self.inputs = np.zeros(n_inputs)
        self.output = 0
        self.delta = 0
    
    def compute(self,inputs):
        return np.dot(self.weights,inputs)
    
    def activate(self,inputs,function):
        self.inputs = np.copy(inputs)
        self.output = function(self.compute(inputs))
        return self.output
    

class Network:
    def __init__(self,layers_sizes,learning_rate,weights=None):
        self.layers = []
        self.learning_rate = learning_rate
            
        for i in range(1,len(layers_sizes)):
            layer = []
            for j in range(layers_sizes[i]):
                if weights is not None:
                    perceptron = Perceptron(layers_sizes[i-1],weights[i][j])
                else:
                    perceptron = Perceptron(layers_sizes[i-1])
                layer.append(perceptron)
            self.layers.append(layer)
        
    def forward_pass(self,raw_inputs):
        inputs = list(raw_inputs)
        inputs.append(1)
        
        for i in range(len(self.layers)):
            outputs = []
            
            for j in range(len(self.layers[i])):
                outputs.append(self.layers[i][j].activate(inputs,sigmoid))
        
            inputs = outputs.copy()
            inputs.append(1)
            
        return outputs
    
    def backward_pass(self,outputs,expected):
        for i in range(len(self.layers)-1,-1,-1):
            layer = self.layers[i]
            if i == len(self.layers)-1:
                delta = outputs - expected
                for j in range(len(layer)):
                    perceptron = layer[j]
                    perceptron.delta = delta[j]
            else:
                next_layer = self.layers[i+1]
                for j in range(len(layer)):
                    perceptron = layer[j]
                    
                    delta = 0
                    for perceptron_next_layer in next_layer:
                        delta += perceptron_next_layer.weights[j]*perceptron_next_layer.delta
                    perceptron.delta = derivative_sigmoid(perceptron.output)*delta
                
    
    def update_weights(self):
        for i in range(len(self.layers)-1,-1,-1):
            layer = self.layers[i]
            for j in range(len(layer)):
                perceptron = layer[j]
                perceptron.weights -= self.learning_rate*perceptron.delta*perceptron.inputs
                    
    def train(self,inputs,expected):
        outputs = self.forward_pass(inputs)
        self.backward_pass(outputs,expected)
        self.update_weights()
        return outputs
            
    def evaluate(self,outputs):
            prediction = np.argmax(outputs)
            return prediction
                            
            
        
        
        

    