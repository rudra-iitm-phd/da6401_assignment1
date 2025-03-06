import numpy as np
from normalisation import UnitNormalisation as unit_normalise
class ActivationFunctions:

  def __init__(self):
    self.unit_normaliser = unit_normalise()

  def get(self, name):
    if name == 'relu':
      return self.relu
    elif name == 'sigmoid':
      return self.sigmoid
    elif name == 'tanh':
      return self.tanh
    elif name == 'softmax':
      return self.softmax

  def relu(self, x):
    x = x.copy()
    return np.maximum(x, 0)

  def sigmoid(self, x):
    x = x.copy()
    # x = self.unit_normaliser.normalize(x) 
    # x = self.unit_normaliser.normalize(x)
    # x = self.unit_normaliser.normalize(x)
    z = -x
    z = z - np.max(z)
    return 1/(1 + np.exp(z))

  def tanh(self, x):
    x = x.copy()
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

  def softmax(self, x):
    x = x.copy()
    x = x - np.max(x) 
    return np.exp(x) / sum(np.exp(x))