import numpy as np
from normalisation import UnitNormalisation as unit_normalise, LogNormalisation as log_normalise, StandardNormal as std_normalise

class ActivationFunctions:

  def __init__(self):
    self.unit_normaliser = unit_normalise()
    self.log_normaliser = log_normalise()
    self.std_normaliser = std_normalise()

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
    z = -x
    z = z - np.max(z)
    z = self.unit_normaliser.normalize(z)*20 - 10
    return 1/(1 + np.exp(z))

  def tanh(self, x):
    x = x.copy()
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

  def softmax(self, x):
    x = x.copy()
    x = x - np.max(x) 
    x = self.unit_normaliser.normalize(x)*20 - 10
    return np.exp(x) / sum(np.exp(x))