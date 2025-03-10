import numpy as np
from activation import ActivationFunctions
from normalisation import UnitNormalisation as unit_normalise
from copy import deepcopy
from preprocess import PreProcessor as pp
from weight_init import Xavier, Random
import shared
class NeuralNetwork:

  def __init__(self, width_of_layers:list, input_size:int, output_size:int, activation_fn:str, output_fn:str, weight_init:str):


    assert weight_init.lower() in ['random', 'xavier']

    

    self.n_hidden = len(width_of_layers)
    self.width_of_layers = width_of_layers
    self.input_size = input_size
    self.output_size = output_size
    self.total_params = 0
    self.activation_functions = ActivationFunctions()
    self.activation_function = self.activation_functions.get(activation_fn)
    self.pp = pp()
    self.output_fn = self.activation_functions.get('softmax')
    shared.acv_fn = activation_fn
    shared.output_fn = 'softmax'
  
   

    self.weight_init = None

    if weight_init.lower() == 'random':
      self.weight_init = Random()
    elif weight_init.lower() == 'xavier':
      self.weight_init = Xavier(self.input_size, self.output_size)

    self.hidden = self.generate_hidden_layers()
    self.params = self.hidden

    self.unit_normaliser = unit_normalise()
    
    
  def generate_hidden_layers(self) -> dict:
    layers = dict()
    for i,j in enumerate(range(self.n_hidden)):
      if i == 0:
        layers[f'hidden{i}'] = { 'w': self.weight_init.initialize((self.width_of_layers[i], self.input_size)), 'b': self.weight_init.initialize((self.width_of_layers[i], 1)), 'h': self.activation_function}
        

      else:
        layers[f'hidden{i}'] = {'w' : self.weight_init.initialize((self.width_of_layers[i], self.width_of_layers[i-1])) , 'b': self.weight_init.initialize((self.width_of_layers[i], 1)), 'h': self.activation_function  }
        

      self.total_params += layers[f'hidden{i}']['w'].shape[0] * layers[f'hidden{i}']['w'].shape[1] + layers[f'hidden{i}']['b'].shape[0]
      
    layers[f'output{self.n_hidden}'] = {'w' : self.weight_init.initialize((self.output_size, self.width_of_layers[-1])) , 'b': self.weight_init.initialize((self.output_size , 1)), 'h': self.output_fn}

    self.total_params += layers[f'output{self.n_hidden}']['w'].shape[0] * layers[f'output{self.n_hidden}']['w'].shape[1] + layers[f'output{self.n_hidden}']['b'].shape[0]

    return layers


  def forward(self, x:np.ndarray):
    x = deepcopy(x)
    x = self.pp.process(x)
    assert x.shape[0] == self.input_size
    
    k = 0
    logits = None
    prev_block = None
    self.params['hidden0']['x'] = x
    for block in list(self.params.keys()):
      if k == 0:
        
        self.params[block]['a'] = self.params[block]['w'].dot(self.params['hidden0']['x']) + self.params[block]['b'] 
         
      else:
        self.params[block]['a'] = self.params[block]['w'].dot(self.params[prev_block]['o']) + self.params[block]['b']
        
      self.params[block]['o'] = self.params[block]['h'](self.params[block]['a'])
      
      k += 1
      prev_block = block

    logits = self.params[prev_block]['o'].copy()

    return self.params, logits

  def infer(self, x:np.ndarray, logits=True):
    l = self.forward(x)[1]
    if logits:
      return l
    else:
      return np.argmax(l, axis = 0)

  def set_params(self, params:dict):
    self.params = deepcopy(params)

  
  def view_model_summary(self):
    print('Model Summary')
    for i,j in enumerate(list(self.hidden.keys())):
      print('---------------------------------------')
      print(f'{j}')

      print(f"weights : {self.hidden[j]['w'].shape}")
      print(f"bias : {self.hidden[j]['b'].shape}")
      if i == self.n_hidden:
        print(f'activation : {shared.output_fn}')
      else:
        print(f'activation : {shared.acv_fn}')
    
    print('---------------------------------------')
    print(f'Total Number of Parameters : {self.total_params}')
