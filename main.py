from neural_network import NeuralNetwork
from optimizer import Optimizer
import numpy as np
from loss_functions import CrossEntropy

nn = NeuralNetwork([32, 64, 128, 32], 128, 10, ['sigmoid', 'sigmoid', 'sigmoid', 'sigmoid'], 'softmax')
# nn.view_model_summary()
x = np.random.normal(size = (128, 1))
nn.params, nn.logits = nn.forward(x)

loss = CrossEntropy(nn)
loss.compute(nn.infer(x), 1)
grads = loss.backpropagate()

optim = Optimizer(nn.params, 1e-3)
optim.update(grads)









# optim = Optimizer(nn.params)
# optim.backpropagate(10, 2, )
