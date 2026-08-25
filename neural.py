import numpy as np
import random
from math import *

class Perceptron:
    def __init__(self,n_inputs):
        self.n = n_inputs
        self.weights = np.zeros(n_inputs+1)
        for i  in range(n_inputs+1):
            self.weights[i] = random.random()
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
    def __init__(self,n_inputs,layers_sizes,learning_rate,activation_function,derivative_function):
        self.n = n_inputs
        self.layers = []
        self.learning_rate = learning_rate
        self.function = activation_function
        self.derivative_function = derivative_function
            
        for i,layer_size in enumerate(layers_sizes):
            layer = []
            for _ in range(layer_size):
                if i == 0:
                    perceptron = Perceptron(n_inputs)
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
                outputs.append(self.layers[i][j].activate(inputs,self.function))
        
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
                    perceptron.delta = self.derivative_function(perceptron.output)*delta
                
    
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
            
    def evaluate(self,inputs):
            outputs = self.forward_pass(inputs)
            prediction = np.argmax(outputs)
            return prediction
                            
            
        
        
        

    