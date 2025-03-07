import numpy as np
from copy import deepcopy
from loss_functions import LossFunction

class Optimizer:
      def __init__(self, *args):
            pass

      def update(self, *args) -> dict:
            raise NotImplemented


class OptimizerConfig:
      def __init__(self):
            pass
      def configure(self, optim:Optimizer, **kwargs):
            args = {k:val for k,val in kwargs.items()}
            if optim == GradientDescent:
                  return GradientDescent(args['lr'])
            elif optim == MomentumGradientDescent:
                  if 'beta' not in args.keys():
                        return MomentumGradientDescent(args['lr'])
                  else : return MomentumGradientDescent(args['lr'], beta = args['beta'])
            elif optim == NAG:
                  if 'beta' not in args.keys():
                        return NAG(args['lr'], args['loss'])
                  else: return NAG(args['lr'], args['loss'], beta = args['beta'])
            elif optim == Adagrad:
                  return Adagrad(args['lr'])



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
      def __init__(self, learning_rate, beta = 0.8):
            
            self.lr = learning_rate
            self.beta = beta 
            self.prev_update = None
            self.t = 0

      def update(self, params, grads:dict):
            self.prev_update = deepcopy(grads)
            params = deepcopy(params)
            if self.t == 0:
                  self.prev_update = {block:{layer:np.zeros_like(params[block][layer]) if layer!= 'h' else 0 for layer in list(params[block].keys())} for block in list(params.keys()) }
            for block in list(params.keys()):
                  for layer in list(params[block].keys()):
                        if layer in ['w', 'b'] :
                              self.prev_update[block][layer] = self.beta * self.prev_update[block][layer] + grads[block][layer]

                              params[block][layer] = params[block][layer] - self.lr * self.prev_update[block][layer]
            self.t += 1

            return params


class NAG(Optimizer):

      def __init__(self, learning_rate, loss_fn:LossFunction, beta = 0.8):
            
            self.lr = learning_rate
            self.beta = beta 
            self.prev_update = None
            self.t = 0
            self.loss_fn = loss_fn

      def update(self, params, grads:dict):
            self.prev_update = deepcopy(grads)
            params = deepcopy(params)
            if self.t == 0:
                  self.prev_update = {block:{layer:np.zeros_like(params[block][layer]) if layer!= 'h' else 0 for layer in list(params[block].keys())} for block in list(params.keys()) }
                  
            lookahead_params = {block:{layer:params[block][layer] - self.beta * self.prev_update[block][layer] if layer!= 'h' else 0 for layer in list(params[block].keys())} for block in list(params.keys()) }

            grad_lookahead = self.loss_fn.backpropagate(lookahead_params)

            for block in list(params.keys()):
                  for layer in list(params[block].keys()):
                        if layer in ['w', 'b'] :

                              self.prev_update[block][layer] = self.beta * self.prev_update[block][layer] + grad_lookahead[block][layer]

                              params[block][layer] = params[block][layer] - self.lr * self.prev_update[block][layer]
            self.t += 1

            return params
            

class Adagrad(Optimizer):

      def __init__(self, learning_rate):
            
            self.lr = learning_rate
             
            self.prev_update = None
            self.t = 0
            

      def update(self, params, grads:dict):
            self.prev_update = deepcopy(grads)
            params = deepcopy(params)
            if self.t == 0:
                  self.prev_update = {block:{layer:np.zeros_like(params[block][layer]) if layer!= 'h' else 0 for layer in list(params[block].keys())} for block in list(params.keys()) }

            for block in list(params.keys()):
                  for layer in list(params[block].keys()):
                        if layer in ['w', 'b'] :

                              self.prev_update[block][layer] = self.prev_update[block][layer] + np.multiply(grads[block][layer], grads[block][layer])

                              params[block][layer] = params[block][layer] -( self.lr / np.sqrt(self.prev_update[block][layer].sum() + 1e-6)) * self.prev_update[block][layer]
            self.t += 1

            return params


      