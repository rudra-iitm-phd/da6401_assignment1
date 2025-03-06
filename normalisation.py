import numpy as np 

class UnitNormalisation:
      def __init__(self):
            pass 
      def normalize(self, x):
            return (x - np.min(x, axis = 0))/(np.max(x, axis = 0) - np.min(x, axis = 0))