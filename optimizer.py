import numpy as np
from copy import deepcopy


class Optimizer:
      def update(self, *args) -> dict:
            raise NotImplemented
class GradientDescent(Optimizer):
      def __init__(self, learning_rate):
            # params = params
            self.lr = learning_rate
            
      def update(self, params, grads:dict):
            params = deepcopy(params)
            for block in list(params.keys()):
                  for layer in list(params[block].keys()):
                        if layer in ['w', 'b']:
                              params[block][layer] = params[block][layer] -  self.lr*grads[block][layer]
            return params

class MomentumGradientDescent(Optimizer):
      def __init__(self, learning_rate, gamma = 0.8):
            
            self.lr = learning_rate
            self.gamma = gamma 
            
            self.t = 0

      def update(self, params, grads:dict):
            self.prev_update = deepcopy(grads)
            params = deepcopy(params)
            if self.t == 0:
                  self.prev_update = {block:{layer:np.zeros_like(params[block][layer]) if layer!= 'h' else 0 for layer in list(params[block].keys())} for block in list(params.keys()) }
            for block in list(params.keys()):
                  for layer in list(params[block].keys()):
                        if layer in ['w', 'b'] :
                              self.prev_update[block][layer] = self.gamma * self.prev_update[block][layer] + self.lr * grads[block][layer]
                              params[block][layer] = params[block][layer] - self.prev_update[block][layer]
            self.t += 1

            return params
            

      