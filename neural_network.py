import numpy as np
from activation import ActivationFunctions

class NeuralNetwork:

  def __init__(self, width_of_layers:list, input_size:int, output_size:int, activation_list:list, output_fn:str):

    self.n_hidden = len(width_of_layers)
    self.width_of_layers = width_of_layers
    self.input_size = input_size
    self.output_size = output_size
    self.total_params = 0
    self.activation_functions = ActivationFunctions()

    self.activations = {f'{activation}{i}' : self.activation_functions.get(activation) for (i,activation) in enumerate(activation_list + [output_fn])}

    self.hidden = self.generate_hidden_layers() 

    self.params = self.init_params(self.hidden)
    
  def generate_hidden_layers(self) -> dict:
    layers = dict()
    for i,j in enumerate(range(self.n_hidden)):
      if i == 0:
        layers[f'hidden{i}'] = { 'w': np.random.normal(size = (self.width_of_layers[i], self.input_size)), 'b': np.random.normal(size = (self.width_of_layers[i],1)), 'h': self.activations[list(self.activations.keys())[i]]}

      else:
        layers[f'hidden{i}'] = {'w' : np.random.normal(size = (self.width_of_layers[i], self.width_of_layers[i-1])), 'b': np.random.normal(size = (self.width_of_layers[i],1)), 'h': self.activations[list(self.activations.keys())[i]]  }

      self.total_params += layers[f'hidden{i}']['w'].shape[0] * layers[f'hidden{i}']['w'].shape[1] + layers[f'hidden{i}']['b'].shape[0]

    layers[f'output{self.n_hidden}'] = {'w' : np.random.normal( size = (self.output_size, self.width_of_layers[-1])), 'b': np.random.normal(size = (self.output_size , 1)), 'h': self.activations[list(self.activations.keys())[self.n_hidden]]}

    self.total_params += layers[f'output{self.n_hidden}']['w'].shape[0] * layers[f'output{self.n_hidden}']['w'].shape[1] + layers[f'output{self.n_hidden}']['b'].shape[0]

    return layers

  def init_params(self, hidden):
    k = 0
    start = np.random.normal(size = (self.input_size,1))
    prev_block = None
    for block in list(hidden.keys()):
      if k == 0:
        hidden[block]['o'] = hidden[block]['h'](np.matmul(hidden[block]['w'], start) + hidden[block]['b'])
        
      else:
        hidden[block]['o'] = hidden[block]['h'](np.matmul(hidden[block]['w'], hidden[prev_block]['o'] ) + hidden[block]['b'])

      prev_block = block
      k += 1
    return hidden

  def get_logits(self, x:np.ndarray):
    assert x.shape == (self.input_size, 1) 
    k = 0
    logits = None
    prev_block = None
    for block in list(self.params.keys()):
      if k == 0:
        self.params[block]['o'] = self.params[block]['h'](np.matmul(self.params[block]['w'], x) + self.params[block]['b'])
        
      else:
        self.params[block]['o'] = self.params[block]['h'](np.matmul(self.params[block]['w'], self.params[prev_block]['o'] ) + self.params[block]['b'])

      logits = self.params[block]['o']

      k += 1
      prev_block = block


    return logits

  
  def view_model_summary(self):
    print('Model Summary')
    for (i,j),k in zip(enumerate(list(self.hidden.keys())), list(self.activations.keys())):
      print('---------------------------------------')
      print(f'{j}')

      print(f"weights : {self.hidden[j]['w'].shape}")
      print(f"bias : {self.hidden[j]['b'].shape}")
      print(f'activation : {k[:-1]}')
    
    print('---------------------------------------')
    print(f'Total Number of Parameters : {self.total_params}')
