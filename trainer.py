import numpy as np 
from neural_network import NeuralNetwork
from optimizer import Optimizer, OptimizerConfig
from loss_functions import LossFunction
from score_metrics import Metrics
import shared
np.seterr(divide = 'ignore') 

class Trainer:
      def __init__(self, train_data:np.ndarray, train_labels:np.ndarray, val_data:np.ndarray, val_labels:np.ndarray, test_data:np.ndarray, test_labels:np.ndarray, logger):

            self.X_train = train_data
            self.y_train = train_labels

            self.X_val = val_data
            self.y_val = val_labels

            self.X_test = test_data
            self.y_test = test_labels

            self.op_config = OptimizerConfig()
            self.logger = logger

      def learn(self, nn:NeuralNetwork, optim:Optimizer, loss_fn:LossFunction, lr:float, batch_size:int, epochs:int, acc_metrics:Metrics, **kwargs):

            nn = nn
            args = {k:v for k,v in kwargs.items()}

            optim = self.op_config.configure(optim, lr = lr, **args)
            loss = loss_fn()
            acc = acc_metrics()

            max_iter = 10

            for epoch in range(epochs):

                  for iter in range(len(self.X_train)//batch_size):

                        if iter > max_iter :
                              break
                  
                        idx = np.random.randint(batch_size, len(self.X_train))

                        x = self.X_train[idx - batch_size : idx]
                        y = self.y_train[idx - batch_size : idx]

                        shared.x = x
                        shared.y = y


                        params, logits = nn.forward(x)
                        loss.compute(logits, y)
                        grads = loss.backpropagate(params)
                        updated_params = optim.update(params, grads)
                        nn.set_params(updated_params)

                  
                  if epoch % 10 == 0 :
                        train_loss = loss.compute(nn.infer(self.X_train), self.y_train)
                        train_accuracy = acc.compute(nn.infer(self.X_train, False), self.y_train)
                        val_accuracy = acc.compute(nn.infer(self.X_val, False), self.y_val)
                        test_accuracy = acc.compute(nn.infer(self.X_test, False), self.y_test)
                        self.verbosity(train_loss, train_accuracy, val_accuracy, test_accuracy)

                        self.logger.log({ 
                              "Accuracy": round(test_accuracy, 2),
                              "Validation Accuracy" : round(val_accuracy, 2),
                              "Train Accuracy" : round(train_accuracy, 2),
                              "Training Loss" : round(train_loss, 2)
                               
                              })

                        class_names = [f"Class {i}" for i in range(10)]
                        cm_table = self.logger.Table(columns=["Actual", "Predicted"])
                        y_pred = nn.infer(self.X_train, False)
                        for true_label, pred_label in zip(self.y_train, y_pred):
                              cm_table.add_data(class_names[true_label], class_names[pred_label])

                        self.logger.log({"Confusion matrix":self.logger.plot.confusion_matrix(probs = None, y_true = self.y_train, preds = y_pred, class_names = class_names)})
                        self.logger.log({"Confusion Matrix Table": cm_table})
                  

      def verbosity(self, train_loss, train_accuracy, val_accuracy, test_accuracy):
            print(f'Loss : {round(train_loss, 2)}      Train accuracy : {round(train_accuracy, 2)}      Validation accuracy : {round(val_accuracy, 2)}      Test Accuracy : {round(test_accuracy, 2)}')
            print('---------------------------------------------------------------------------------------------------------')
            








