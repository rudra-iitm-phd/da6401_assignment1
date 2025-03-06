import numpy as np

class WeightInitializer:

      def __init__(self):
            pass

      def initialize(self, *args)->np.ndarray:
            raise NotImplemented


class Random(WeightInitializer):

      def __init__(self):
            pass 

      def initialize(self, dimension:tuple) -> np.ndarray:
            return np.random.rand(dimension[0], dimension[1]) - 0.5


class Xavier(WeightInitializer):

      def __init__(self, n_input:int, n_output:int):

            self.r = np.sqrt(6 / (n_input + n_output))

      def initialize(self, dimension:tuple) -> np.ndarray:

            uniform = np.random.rand(dimension[0], dimension[1])

            scaled_uniform = (uniform - uniform.min())/(uniform.max() - uniform.min())*(2*self.r) - self.r

            return scaled_uniform
            