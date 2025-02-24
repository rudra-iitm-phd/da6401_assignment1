import numpy as np

class ActivationFunctions:

  def __init__(self):
    pass

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
    return max(0, x)

  def sigmoid(self, x):
    return 1/(1 + np.exp(-x))

  def tanh(self, x):
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

  def softmax(self, x):
    return np.exp(x)/sum(np.exp(x))