import numpy as np
from copy import deepcopy
from loss_functions import LossFunction
import shared

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
            if 'weight_decay' in args.keys():
                  weight_decay = args['weight_decay']
            else: weight_decay = 0

            if optim == GradientDescent:
                  return GradientDescent(args['lr'], weight_decay = weight_decay)
                  
            elif optim == MomentumGradientDescent:
                  if 'beta' not in args.keys():
                        return MomentumGradientDescent(args['lr'], weight_decay = weight_decay)
                  else : return MomentumGradientDescent(args['lr'], beta = args['beta'], weight_decay = weight_decay)
            elif optim == NAG:
                  if 'beta' not in args.keys():
                        return NAG(learning_rate = args['lr'], loss_fn =  args['loss'], forward = args['forward'], weight_decay = weight_decay)
                  else: return NAG(learning_rate = args['lr'], loss_fn =  args['loss'], forward = args['forward'], beta = args['beta'], weight_decay = weight_decay)

            elif optim == Adagrad:

                  return Adagrad(args['lr'], weight_decay = weight_decay)

            elif optim == RMSProp:

                  return RMSProp(args['lr'], beta = args['beta'], weight_decay = weight_decay)



class GradientDescent(Optimizer):
      def __init__(self, learning_rate, weight_decay = 0):
            # params = params
            self.lr = learning_rate
            self.weight_decay = weight_decay
            
      def update(self, params, grads:dict):
            params = deepcopy(params)
            for block in list(params.keys()):
                  for layer in list(params[block].keys()):
                        if layer == 'w':
                              params[block][layer] = params[block][layer] -  self.lr*(grads[block][layer] + self.weight_decay * params[block][layer])
                        elif layer == 'b':
                              params[block][layer] = params[block][layer] -  self.lr*(grads[block][layer])

            return params

class MomentumGradientDescent(Optimizer):
      def __init__(self, learning_rate, weight_decay = 0, beta = 0.8):
            
            self.lr = learning_rate
            self.beta = beta 
            self.prev_update = None
            self.t = 0
            self.weight_decay = weight_decay

      def update(self, params, grads:dict):
            self.prev_update = deepcopy(grads)
            params = deepcopy(params)
            if self.t == 0:
                  self.prev_update = {block:{layer:np.zeros_like(params[block][layer]) if layer!= 'h' else 0 for layer in list(params[block].keys())} for block in list(params.keys()) }
            for block in list(params.keys()):
                  for layer in list(params[block].keys()):
                        if layer in ['w', 'b'] :
                              self.prev_update[block][layer] = self.beta * self.prev_update[block][layer] + grads[block][layer]
                              if layer == 'w':
                                    params[block][layer] = params[block][layer] - self.lr * (self.prev_update[block][layer] + self.weight_decay * params[block][layer])
                              elif layer == 'b':
                                    params[block][layer] = params[block][layer] - self.lr * (self.prev_update[block][layer] )
            self.t += 1

            return params


class NAG(Optimizer):

      def __init__(self, learning_rate,  forward, loss_fn:LossFunction, weight_decay = 0,  beta = 0.8):
            
            self.lr = learning_rate
            self.beta = beta 
            self.prev_update = None
            self.t = 0
            self.loss_fn = loss_fn()
            self.forward = forward
            self.weight_decay = 0
            

      def update(self, params, grads:dict):
            # self.prev_update = deepcopy(grads)
            x = shared.x
            y = shared.y
            params = deepcopy(params)
            if self.t == 0:
                  self.prev_update = {block:{layer:np.zeros_like(params[block][layer]) if layer!= 'h' else 0 for layer in list(params[block].keys())} for block in list(params.keys()) }
                  
            lookahead_params = {block:{layer:params[block][layer] - self.beta * self.prev_update[block][layer] if layer!= 'h' else 0 for layer in list(params[block].keys())} for block in list(params.keys()) }

            _, logits = self.forward(x)
            self.loss_fn.compute(logits, y)
            grad_lookahead = self.loss_fn.backpropagate(lookahead_params)

            for block in list(params.keys()):
                  for layer in list(params[block].keys()):
                        if layer in ['w', 'b'] :

                              self.prev_update[block][layer] = self.beta * self.prev_update[block][layer] + grad_lookahead[block][layer]
                              
                              if layer == 'w':
                                    params[block][layer] = params[block][layer] - self.lr * (self.prev_update[block][layer] + self.weight_decay * params[block][layer])
                              elif layer == 'b':
                                    params[block][layer] = params[block][layer] - self.lr * (self.prev_update[block][layer])


            self.t += 1

            return params
            

class Adagrad(Optimizer):

      def __init__(self, learning_rate, weight_decay = 0):
            
            self.lr = learning_rate
             
            self.prev_update = None
            self.t = 0
            self.weight_decay = weight_decay
            

      def update(self, params, grads:dict):
            
            params = params.copy()
            if self.t == 0:
                  self.prev_update = {block:{layer:np.zeros_like(params[block][layer]) if layer!= 'h' else 0 for layer in list(params[block].keys())} for block in list(params.keys()) }

            for block in list(params.keys()):
                  for layer in list(params[block].keys()):
                        if layer in ['w', 'b'] :
                              
                              self.prev_update[block][layer] +=  np.power(grads[block][layer], 2)
                              
                              if layer == 'w':
                                    params[block][layer] = params[block][layer] - np.multiply(self.lr/np.sqrt(self.prev_update[block][layer] + 1e-8), self.prev_update[block][layer]) - np.multiply(self.lr/np.sqrt(self.prev_update[block][layer] + 1e-8), params[block][layer])
                              

                              
            self.t += 1

            return params

class RMSProp(Optimizer):

      def __init__(self, learning_rate, beta:float, weight_decay = 0):
            
            self.lr = learning_rate
             
            self.prev_update = None
            self.t = 0
            self.weight_decay = weight_decay
            self.beta = beta
            

      def update(self, params, grads:dict):
            
            params = params.copy()
            if self.t == 0:
                  self.prev_update = {block:{layer:np.zeros_like(params[block][layer]) if layer!= 'h' else 0 for layer in list(params[block].keys())} for block in list(params.keys()) }

            for block in list(params.keys()):
                  for layer in list(params[block].keys()):
                        if layer in ['w', 'b'] :
                              
                              self.prev_update[block][layer] =  self.beta * self.prev_update[block][layer] + (1-self.beta) * np.power(grads[block][layer], 2)
                              
                              if layer == 'w':
                                    params[block][layer] = params[block][layer] - np.multiply(self.lr/np.sqrt(self.prev_update[block][layer]+1e-8), grads[block][layer]) - self.weight_decay*self.lr*params[block][layer]

                              elif layer == 'b':
                                    params[block][layer] = params[block][layer] - np.multiply(self.lr/np.sqrt(self.prev_update[block][layer]+1e-8), self.prev_update[block][layer])
                              

                              
            self.t += 1

            return params


      