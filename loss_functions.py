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
            self.pred_logits = None
            

      def compute(self, pred_logits:np.ndarray, true_classes:np.ndarray):
            
            self.pred_logits = pred_logits
            # converting the true class into a one-hot vector
            self.actual_class_vector = np.zeros_like(pred_logits)
            self.loss = 0
            for m in range(self.actual_class_vector.shape[1]):
                  self.actual_class_vector[true_classes[m], m] = 1
                  self.loss -= np.log(pred_logits[true_classes[m],m])

            #computing the cross entropy loss
            self.loss = self.loss / self.actual_class_vector.shape[1]
            return self.loss

      def backpropagate(self):
            assert self.params['hidden0']['x'].shape[0] == self.params['hidden0']['w'].shape[1]
            prev_layer = 'loss'
            

            e_l = self.actual_class_vector
            


            grads = {block:{layer:np.zeros(shape = (self.params[block][layer].shape[0], self.params[block][layer].shape[1])) if layer!= 'h' else 0 for layer in list(self.params[block].keys())} for block in list(self.params.keys()) }

            for block in reversed((list(self.params.keys()))):
                  k = 0
                  # print(block)
                  for layer in reversed((list(self.params[block].keys()))):
                        
                        if block[:-1] == 'output' :
                              if  layer == 'o':

                                    grads[block][layer] = np.multiply(e_l, - 1/self.pred_logits).mean(axis  =-1).reshape(-1, 1)

                              elif layer == 'a':
                              
                                    grads[block]['a'] = -(e_l - self.pred_logits).mean(axis = -1).reshape(-1, 1)
                        
                        elif block[:-1] != 'output' and layer == 'a':
                              
                              grads[block][layer] = np.multiply(grads[block]['o'], self.g_prime(self.params[block]['o']).mean(axis = -1).reshape(-1, 1)).mean(axis = -1).reshape(-1, 1)

                        elif block[:-1] != 'output' and layer == 'o':
                              
                              grads[block]['o'] = np.multiply(grads[prev_block]['w'].T, grads[prev_block]['a'].mean(axis = -1) ).mean(axis = -1).reshape(-1, 1)
                              
                        elif layer == 'w' and int(block[-1]) - 1 >= 0:
                              
                              # print(grads[block]['a'].mean(axis = -1).reshape(-1, 1).shape, self.params[list(self.params.keys())[int(block[-1]) - 1 ]]['o'].T.shape)

                              grads[block][layer] = np.matmul(grads[block]['a'].mean(axis = -1).reshape(-1, 1), self.params[list(self.params.keys())[int(block[-1]) - 1 ]]['o'].mean(axis = -1).reshape(-1, 1).T)

                        elif layer == 'w' and int(block[-1]) == 0 :
                              
                              grads[block][layer] = np.matmul(grads[block]['a'].reshape(-1, 1), self.params['hidden0']['x'].mean(axis = -1).reshape(-1, 1).T)

                        elif layer == 'b':
                              grads[block][layer] = grads[block]['a']

                        # if layer != 'h':
                        #       print(f'{layer} : {grads[block][layer].shape}')

                        # if layer != 'h':
                        #       print(grads[block][layer].shape == self.params[block][layer].shape)

                              

                        prev_block = block
                        k += 1
            return grads

      def g_prime(self, x):
            return self.activations['sigmoid'](x)*(1 - self.activations['sigmoid'](x))


