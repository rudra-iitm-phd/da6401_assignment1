from neural_network import NeuralNetwork
from optimizer import Optimizer
import numpy as np

nn = NeuralNetwork([32, 64, 128], 128, 10, ['sigmoid', 'sigmoid', 'sigmoid'], 'softmax')
nn.view_model_summary()


optim = Optimizer(nn.params)
optim.backpropagate(10, 2, np.random.normal(size = (128, 1)))
