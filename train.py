import numpy as np 
from neural_network import NeuralNetwork
from optimizer import Optimizer
from loss_functions import LossFunction
from score_metrics import Metrics

class Trainer:
      def __init__(self, train_data:np.ndarray, train_labels:np.ndarray, val_data:np.ndarray, val_labels:np.ndarray):

            self.X_train = train_data
            self.y_train = train_labels

            self.X_val = val_data
            self.y_val = val_labels

      def learn(self, nn:NeuralNetwork, optim:Optimizer, loss_fn:LossFunction, lr:float, batch_size:int, epochs:int, acc_metrics:Metrics):

            nn = nn 
            optim = optim(lr)
            loss = loss_fn()
            acc = acc_metrics()

            for epoch in range(epochs):
                  
                  idx = np.random.randint(batch_size, len(self.X_train))

                  x = self.X_train[idx - batch_size : idx]
                  y = self.y_train[idx - batch_size : idx]

                  params, logits = nn.forward(x)
                  loss.compute(logits, y)
                  grads = loss.backpropagate(params)
                  updated_params = optim.update(params, grads)
                  nn.set_params(updated_params)

                  if epoch % 50 == 0 :
                        train_loss = loss.compute(nn.infer(x), y)
                        train_accuracy = acc.compute(nn.infer(x, False), y)
                        val_accuracy = acc.compute(nn.infer(self.X_val, False), self.y_val)
                        self.verbosity(train_loss, train_accuracy, val_accuracy)
                  

      def verbosity(self, train_loss, train_accuracy, val_accuracy):
            print(f'Loss : {round(train_loss, 2)}      Train accuracy : {round(train_accuracy, 2)}      Validation accuracy : {round(val_accuracy, 2)}')
            print('-------------------------------------------------------------------------------------------------------------------------------------')
            








