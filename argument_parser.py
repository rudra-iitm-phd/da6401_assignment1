import argparse

parser = argparse.ArgumentParser(description = "Train the neural network for the classification of Fashion MNIST dataset")

parser.add_argument('-e', '--epochs', 
                  type = int, default = 1000, 
                  help = 'Number of epochs to train neural network')

parser.add_argument('-b', '--batch_size', 
                  type = int, default = 512, 
                  help = 'Batch size used to train neural network.')

parser.add_argument('-l', '--loss', 
                  type = str, default = 'cross_entropy', 
                  help = 'choices: ["mean_squared_error", "cross_entropy"]')

parser.add_argument('-o', '--optimizer', 
                  type = str, default = 'momentum', 
                  help = 'choices: ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]')

parser.add_argument('-lr', '--learning_rate', 
                  type = float, default = 1e-2, 
                  help = 'Learning rate used to optimize model parameters')

parser.add_argument('-w_i', '--weight_init', 
                  type = str, default = 'random', 
                  help = 'choices : ["random", "xavier"]')

parser.add_argument('-hl', '--hidden_layers', 
                  type = int, default = 728, nargs = '+', 
                  help = 'Hidden layer configuration e.g. : [32, 128, 64]')

parser.add_argument('-si', '--input_size', 
                  type = int, default = 728, 
                  help = 'Size of the input')

parser.add_argument('-so', '--output_size', 
                  type = int, default = 10, 
                  help = 'Output size')

parser.add_argument('-beta', '--beta', 
                  type = float, default = 0.9, 
                  help = 'Value of beta used in rmsprop')


parser.add_argument('-a', '--activation', 
                  type = str, default = 'relu', nargs = '+',
                  help = 'Choices : ["identity", "sigmoid", "tanh", "ReLU"]')

parser.add_argument('-w_d', '--weight_decay', 
                  type = float, default = 0,
                  help = 'Weight decay used by optimizers.(L2 Regularization)')






