import numpy as np 

class UnitNormalisation:
      def __init__(self):
            pass 
      def normalize(self, x):
            x = x.copy()
            x = np.clip(x, -1e10, 1e10)
            return (x - np.min(x, axis = 0))/(np.max(x, axis = 0) - np.min(x, axis = 0))

class LogNormalisation:
      def __inti__(self):
            pass
      def normalize(self, x):
            return np.log10(x)

class StandardNormal:
      def __init__(self):
            pass 
      def normalize(self, x:np.ndarray):
            return (x - x.mean(axis = 0))/x.std()