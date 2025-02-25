import numpy as np
from activation import ActivationFunctions
from neural_network import NeuralNetwork

class Optimizer:
      def __init__(self, params:dict, learning_rate):
            self.params = params
            self.lr = learning_rate
            
      def update(self, grads:dict):
            for block in list(self.params.keys()):
                  for layer in list(self.params[block].keys()):
                        if layer != 'h':
                              self.params[block][layer] -= self.lr*grads[block][layer]
            