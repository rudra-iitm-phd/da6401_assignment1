import numpy as np
from neural_network import NeuralNetwork
from activation import ActivationFunctions

class CrossEntropy:
      def __init__(self, nn:NeuralNetwork):
            self.loss = None
            self.nn = nn
            self.params = nn.params
            self.activation_functions = ActivationFunctions()
            self.activations = {f : self.activation_functions.get(f) for f in ['sigmoid', 'relu', 'tanh', 'softmax']}
            

      def compute(self, pred_logits:np.ndarray, true_class:int):

            # converting the true class into a one-hot vector
            self.actual_class_vector = np.zeros_like(pred_logits)
            self.actual_class_vector[true_class] = 1

            #computing the cross entropy loss
            self.loss = -np.log(pred_logits[true_class])
            return self.loss

      def backpropagate(self):
            assert self.params['hidden0']['x'].shape[0] == self.params['hidden0']['w'].shape[1]
            prev_layer = 'loss'
            

            e_l = self.actual_class_vector


            grads = {block:{layer:np.zeros_like(self.params[block][layer]) if layer!= 'h' else 0 for layer in list(self.params[block].keys())} for block in list(self.params.keys()) }

            for block in reversed((list(self.params.keys()))):
                  k = 0
                  
                  for layer in reversed((list(self.params[block].keys()))):
                        
                        
                        if block[:-1] == 'output' and layer == 'o':
                              
                              grads[block]['a'] = -(e_l - self.params[block][layer])
                              
                        
                        elif layer == 'h':
                              grads[block][layer] = np.matmul(self.params[block]['w'].T, grads[block]['a'])

                        elif layer == 'o':
                              grads[block]['a'] = np.multiply(grads[prev_block]['h'], self.g_prime(self.params[block]['o']))
                              

                        elif layer == 'w' and int(block[-1]) - 1 >= 0:
                              grads[block][layer] = np.multiply(grads[block]['a'], self.params[list(self.params.keys())[max(0, int(block[-1]) - 1)]]['o'].T)

                              
                              
                        elif layer == 'w' and int(block[-1]) == 0 :
                              grads[block][layer] = np.multiply(grads[block]['a'], self.params['hidden0']['x'].T)

                        # if layer != 'h':
                        #       print(grads[block][layer].shape == self.params[block][layer].shape)

                              

                        prev_block = block
                        k += 1
            return grads

      def g_prime(self, x):
            return self.activations['sigmoid'](x)*(1 - self.activations['sigmoid'](x))


