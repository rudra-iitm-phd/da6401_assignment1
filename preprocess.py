import numpy as np 

class PreProcessor:
      def __init__(self):
            pass
      def process(self, x):
            batch_size = x.shape[0]
            data = x.transpose((1, 2, 0)).reshape(-1, batch_size)
            data = data / np.max(data)
            return data
            