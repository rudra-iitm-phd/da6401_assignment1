import numpy as np
from neural_network import NeuralNetwork
from activation import ActivationFunctions
from copy import deepcopy
from normalisation import UnitNormalisation as unit_normalise
import shared



class LossFunction:
      def compute(self, *args)->float:
            raise NotImplemented

      def backpropagate(self, *args)->dict:
            raise NotImplemented

class CrossEntropy(LossFunction):
      def __init__(self):
            self.loss = None
            # self.nn = nn
            # params = nn.params
            self.activation_functions = ActivationFunctions()
            self.activations = {f : self.activation_functions.get(f) for f in ['sigmoid', 'relu', 'tanh', 'softmax']}
            self.pred_logits = None
            self.unit_normaliser = unit_normalise()


            
            

      def compute(self, pred_logits:np.ndarray, true_classes:np.ndarray):
            
            self.pred_logits = pred_logits
            # converting the true class into a one-hot vector
            self.actual_class_vector = np.zeros_like(pred_logits)
            
            self.loss = 0
            self.del_yl = self.actual_class_vector.copy()
            
            for m in range(self.actual_class_vector.shape[1]):

                  self.actual_class_vector[:, m][true_classes[m]] = 1
                  self.loss -= np.log10(pred_logits[true_classes[m],m])
                  one_hot_class = self.actual_class_vector[:, m]
                  self.del_yl[:, m] = -one_hot_class/pred_logits[true_classes[m],m]

            return self.loss / pred_logits.shape[-1]

      def backpropagate(self, params):
            params = deepcopy(params)
            assert params['hidden0']['x'].shape[0] == params['hidden0']['w'].shape[1]
            prev_layer = 'loss'
            

            e_l = self.actual_class_vector
            


            grads = {block:{layer:np.zeros(shape = (params[block][layer].shape[0], params[block][layer].shape[1])) if layer!= 'h' else 0 for layer in list(params[block].keys())} for block in list(params.keys()) }

            for block in reversed((list(params.keys()))):
                  k = 0
                  
                  for layer in reversed((list(params[block].keys()))):
                        
                        if block[:-1] == 'output' :
                              if  layer == 'o':

                                    grads[block][layer] = self.del_yl.copy()
                                    

                              elif layer == 'a':

                                    grads[block]['a'] = -(e_l - self.pred_logits)
                        
                        elif block[:-1] != 'output' and layer == 'a':
                              
                              grads[block][layer] = np.multiply(np.clip(grads[block]['o'], -1e10, 1e10), self.g_prime(params[block]['a'], shared.acv_fn))*(1/self.pred_logits.shape[-1])

                        elif block[:-1] != 'output' and layer == 'o':
                              grads[block]['o'] = np.dot(grads[prev_block]['w'].T,(grads[prev_block]['a']))*(1/self.pred_logits.shape[-1])
                              

                        if layer == 'w' and int(block[-1]) > 0:
                              grads[block][layer] = np.dot(grads[block]['a'], params[list(params.keys())[int(block[-1]) - 1 ]]['o'].T)*(1/self.pred_logits.shape[-1])
                              

                        elif layer == 'w' and int(block[-1]) == 0 :
                              grads[block][layer] = np.dot(grads[block]['a'], params['hidden0']['x'].T)*(1/self.pred_logits.shape[-1])
                              

                        elif layer == 'b':
                              grads[block][layer] = grads[block]['a'].copy()
                              grads[block][layer] = grads[block][layer].sum(axis = -1).reshape(-1, 1)*(1/self.pred_logits.shape[-1])

                       
                        prev_block = block
                        k += 1
            return grads

      def g_prime(self, x, activation):
            x = x.copy()
            # return 
            if activation == 'relu':
                  return x>0
            elif activation == 'sigmoid':
                  return np.multiply(self.activations['sigmoid'](x), (1 - self.activations['sigmoid'](x)))


class MeanSquaredError(LossFunction):
      def __init__(self):
            self.loss = None
            # self.nn = nn
            # params = nn.params
            self.activation_functions = ActivationFunctions()
            self.activations = {f : self.activation_functions.get(f) for f in ['sigmoid', 'relu', 'tanh', 'softmax']}
            self.pred_logits = None
            self.unit_normaliser = unit_normalise()


            
            

      def compute(self, pred_logits:np.ndarray, true_classes:np.ndarray):
            
            self.pred_logits = pred_logits
            self.actual_class_vector = np.zeros_like(pred_logits)
            
            self.loss = 0
            self.del_yl = self.actual_class_vector.copy()
            
            for m in range(self.actual_class_vector.shape[1]):
                  self.actual_class_vector[:, m][true_classes[m]] = 1

            self.loss = np.power(self.actual_class_vector - pred_logits, 2)
            
            return self.loss.sum(axis = -1).sum(axis = -1)/(pred_logits.shape[1]*pred_logits.shape[0])

      def backpropagate(self, params):
            params = deepcopy(params)
            assert params['hidden0']['x'].shape[0] == params['hidden0']['w'].shape[1]
            prev_layer = 'loss'
            

            e_l = self.actual_class_vector
            


            grads = {block:{layer:np.zeros(shape = (params[block][layer].shape[0], params[block][layer].shape[1])) if layer!= 'h' else 0 for layer in list(params[block].keys())} for block in list(params.keys()) }

            for block in reversed((list(params.keys()))):
                  k = 0
                  
                  for layer in reversed((list(params[block].keys()))):
                        
                        if block[:-1] == 'output' :
                              if  layer == 'o':

                                    grads[block][layer] = -(e_l - self.pred_logits)*2
                                    

                              elif layer == 'a':

                                    grads[block]['a'] = np.multiply(grads[block]['o'], self.pred_logits)*2
                        
                        elif block[:-1] != 'output' and layer == 'a':
                              
                              grads[block][layer] = np.multiply(np.clip(grads[block]['o'], -1e10, 1e10), self.g_prime(params[block]['a'], shared.acv_fn))*(1/self.pred_logits.shape[-1])

                        elif block[:-1] != 'output' and layer == 'o':
                              grads[block]['o'] = np.dot(grads[prev_block]['w'].T,(grads[prev_block]['a']))*(1/self.pred_logits.shape[-1])
                              

                        if layer == 'w' and int(block[-1]) > 0:
                              grads[block][layer] = np.dot(grads[block]['a'], params[list(params.keys())[int(block[-1]) - 1 ]]['o'].T)*(1/self.pred_logits.shape[-1])
                              

                        elif layer == 'w' and int(block[-1]) == 0 :
                              grads[block][layer] = np.dot(grads[block]['a'], params['hidden0']['x'].T)*(1/self.pred_logits.shape[-1])
                              

                        elif layer == 'b':
                              grads[block][layer] = grads[block]['a'].copy()
                              grads[block][layer] = grads[block][layer].sum(axis = -1).reshape(-1, 1)*(1/self.pred_logits.shape[-1])

                       
                        prev_block = block
                        k += 1
            return grads

      def g_prime(self, x, activation):
            x = x.copy()
            if activation == 'relu':
                  return x>0
            elif activation == 'sigmoid':
                  return np.multiply(self.activations['sigmoid'](x), (1 - self.activations['sigmoid'](x)))

            


