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

parser.add_argument('-sz', '--hidden_size', 
                  type = int, default = [784], nargs = '+', 
                  help = 'Number of hidden neurons in a feedforward layer.')

parser.add_argument('-nhl', '--num_layers', 
                  type = int, default = 1, 
                  help = 'Number of hidden layers used in feedforward neural network.')

parser.add_argument('-si', '--input_size', 
                  type = int, default = 784, 
                  help = 'Size of the input')

parser.add_argument('-so', '--output_size', 
                  type = int, default = 10, 
                  help = 'Output size')

parser.add_argument('-beta', '--beta', 
                  type = float, default = 0.5, 
                  help = 'Value of beta used in rmsprop')


parser.add_argument('-a', '--activation', 
                  type = str, default = 'relu',
                  help = 'Choices : ["identity", "sigmoid", "tanh", "ReLU"]')

parser.add_argument('-w_d', '--weight_decay', 
                  type = float, default = 0,
                  help = 'Weight decay used by optimizers.(L2 Regularization)')

parser.add_argument('-beta1', '--beta1', 
                  type = float, default = 0.5,
                  help = 'Beta1 used by adam and nadam optimizers')


parser.add_argument('-beta2', '--beta2', 
                  type = float, default = 0.5,
                  help = 'Beta2 used by adam and nadam optimizers')

parser.add_argument('-eps', '--epsilon', 
                  type = float, default = 1e-6,
                  help = 'Epsilon used by optimizers')

parser.add_argument('-m', '--momentum', 
                  type = float, default = 0.5,
                  help = 'Momentum used by momentum and nag optimizers.')

parser.add_argument('-we', '--wandb_entity', 
                  type = str, default = 'da24d008-iit-madras' ,
                  help = 'Wandb Entity used to track experiments in the Weights & Biases dashboard')

parser.add_argument('-wp', '--wandb_project', 
                  type = str, default = 'da24d008-assignment1' ,
                  help = 'Project name used to track experiments in Weights & Biases dashboard')

parser.add_argument('--wandb_sweep', action='store_true', help='Enable W&B sweep')

parser.add_argument('--sweep_id', type = str, help = "Sweep ID", default = None)

parser.add_argument('-d', '--dataset', type = str, default = 'fashion_mnist', help = 'Dataset choices: ["mnist", "fashion_mnist"]')
