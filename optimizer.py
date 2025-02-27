import numpy as np
from activation import ActivationFunctions
from neural_network import NeuralNetwork

class GradientDescent:
      def __init__(self, params:dict, learning_rate):
            self.params = params
            self.lr = learning_rate
            
      def update(self, grads:dict):
            for block in list(self.params.keys()):
                  for layer in list(self.params[block].keys()):
                        if layer in ['w', 'b']:
                              self.params[block][layer] -= self.lr*grads[block][layer]
            return self.params

class MomentumGradientDescent:
      def __init__(self, params:dict, learning_rate, gamma = 0.8):
            self.params = params
            self.lr = learning_rate
            self.gamma = gamma 
            self.prev_update = {block:{layer:np.zeros_like(self.params[block][layer]) if layer!= 'h' else 0 for layer in list(self.params[block].keys())} for block in list(self.params.keys()) }
            self.t = 0

      def update(self, grads:dict):
            self.prev_update = grads.copy()

            for block in list(self.params.keys()):
                  for layer in list(self.params[block].keys()):
                        if layer in ['w', 'b'] :
                              self.prev_update[block][layer] = self.gamma * self.prev_update[block][layer] + self.lr * grads[block][layer]
                              self.params[block][layer] = self.params[block][layer] - self.prev_update[block][layer]
            self.t += 1

            return self.params
            

      