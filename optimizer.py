import numpy as np
from activation import ActivationFunctions

class Optimizer:
      def __init__(self, params:dict):
            self.params = params
            self.acv = ActivationFunctions()
            self.activations = {f : self.acv.get(f) for f in ['sigmoid', 'relu', 'tanh', 'softmax']}
            
            

      def backpropagate(self, loss, true_class, features:np.ndarray):

            assert features.shape[0] == self.params['hidden0']['w'].shape[1]
            prev_layer = 'loss'
            

            e_l = np.zeros_like(self.params[f'output{len(list(self.params.keys()))-1}']['o'])

            e_l[true_class] = 1

            grads = {block:{layer:0 for layer in list(self.params[block].keys())} for block in list(self.params.keys()) }

            for block in reversed((list(self.params.keys()))):
                  k = 0
                  
                  for layer in reversed((list(self.params[block].keys()))):
                        
                        
                        if block[:-1] == 'output' and layer == 'o':
                              
                              grads[block]['a'] = -(e_l - self.params[block][layer])
                              
                        
                        elif layer == 'h':
                              grads[block][layer] = np.matmul(self.params[block]['w'].T, grads[block]['a'])

                        elif layer == 'o':
                              grads[block]['a'] = np.multiply(grads[prev_block]['h'], self.g_prime(self.params[block]['o']))
                              

                        elif layer == 'w' and int(block[-1]) - 1 > 0:
                              grads[block][layer] = np.multiply(grads[block]['a'], self.params[list(self.params.keys())[max(0, int(block[-1]) - 1)]]['o'].T)
                              
                        elif layer == 'w' and int(block[-1]) - 1 == 0 :
                              grads[block][layer] = np.multiply(grads[block]['a'], features.T)



                              

                        prev_block = block
                        k += 1


      def g_prime(self, x):
            return self.activations['sigmoid'](x)*(1 - self.activations['sigmoid'](x))