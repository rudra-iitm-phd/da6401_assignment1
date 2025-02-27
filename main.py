# from neural_network import NeuralNetwork
# from optimizer import MomentumGradientDescent as mgd
import numpy as np
# from loss_functions import CrossEntropy
from configure import Configure

# configure network

configuration = Configure()
# configuration.show_blueprint()
configuration_script = {
      'hidden_layers':[32, 64, 128],
      'input_size':128, 
      'output_size':10,
      'activation_functions':['sigmoid', 'sigmoid', 'sigmoid'],
      'optimizer':'vanilla',
      'loss_function':'cross entropy'
}
nn, optim, loss_fn = configuration.configure(configuration_script)

# nn = NeuralNetwork([32, 64, 128, 32], 128, 10, ['sigmoid', 'sigmoid', 'sigmoid', 'sigmoid'], 'softmax')
x = np.random.normal(size = (128, 1))
nn.view_model_summary()

#forward data to network
nn.params, nn.logits = nn.forward(x)

#initialize loss
loss = loss_fn(nn)

for i in range(1000):
      
      

      #compute loss
      loss.compute(nn.infer(x), np.array([np.random.randint(0,10) for _ in range(1)]))


      #backpropagate loss
      grads = loss.backpropagate()

      #initialize optimizer
      op = optim(nn.params, 1e-3)

      #update the params
      updated_params = op.update(grads)

      #pass the updated params to neural net
      nn.set_params(updated_params)

      









# optim = Optimizer(nn.params)
# optim.backpropagate(10, 2, )
