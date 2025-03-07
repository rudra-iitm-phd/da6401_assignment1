import numpy as np
from neural_network import NeuralNetwork
from activation import ActivationFunctions
from copy import deepcopy
from normalisation import UnitNormalisation as unit_normalise



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
                  
            #computing the cross entropy loss
            

            # print(self.del_yl)
            # print(true_classes)
            # print(pred_logits)
            # print(self.actual_class_vector)

            return self.loss / pred_logits.shape[-1]

      def backpropagate(self, params):
            params = deepcopy(params)
            assert params['hidden0']['x'].shape[0] == params['hidden0']['w'].shape[1]
            prev_layer = 'loss'
            

            e_l = self.actual_class_vector
            


            grads = {block:{layer:np.zeros(shape = (params[block][layer].shape[0], params[block][layer].shape[1])) if layer!= 'h' else 0 for layer in list(params[block].keys())} for block in list(params.keys()) }

            for block in reversed((list(params.keys()))):
                  k = 0
                  # print(block)
                  for layer in reversed((list(params[block].keys()))):
                        
                        if block[:-1] == 'output' :
                              if  layer == 'o':

                                    grads[block][layer] = self.del_yl.copy()
                                    

                              elif layer == 'a':

                                    grads[block]['a'] = -(e_l - self.pred_logits)
                        
                        elif block[:-1] != 'output' and layer == 'a':
                              # some problem is here
                              # print(params[block]['a'])
                              grads[block][layer] = np.multiply(grads[block]['o'], self.g_prime(params[block]['a']))*(1/self.pred_logits.shape[-1])

                        elif block[:-1] != 'output' and layer == 'o':
                              grads[block]['o'] = np.dot(grads[prev_block]['w'].T,(grads[prev_block]['a']))*(1/self.pred_logits.shape[-1])
                              # grads[block]['o'] = np.matmul(grads[prev_block]['w'].T, grads[prev_block]['a'])

                        if layer == 'w' and int(block[-1]) > 0:
                              grads[block][layer] = np.dot(grads[block]['a'], params[list(params.keys())[int(block[-1]) - 1 ]]['o'].T)*(1/self.pred_logits.shape[-1])
                              # grads[block][layer] = np.matmul(grads[block]['a'], params[list(params.keys())[int(block[-1]) - 1 ]]['o'].T)

                        elif layer == 'w' and int(block[-1]) == 0 :
                              grads[block][layer] = np.dot(grads[block]['a'], params['hidden0']['x'].T)*(1/self.pred_logits.shape[-1])
                              # grads[block][layer] = np.matmul(grads[block]['a'], params['hidden0']['x'].T)

                        elif layer == 'b':
                              grads[block][layer] = grads[block]['a'].copy()
                              grads[block][layer] = grads[block][layer].sum(axis = -1).reshape(-1, 1)*(1/self.pred_logits.shape[-1])

                        # if layer != 'h':
                        #       print(f'{layer} : {grads[block][layer]}')

                        prev_block = block
                        k += 1
            return grads

      def g_prime(self, x):
            x = x.copy()
            # return np.multiply(self.activations['sigmoid'](x), (1 - self.activations['sigmoid'](x)))
            return x>0
            


