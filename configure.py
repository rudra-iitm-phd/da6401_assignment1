from neural_network import NeuralNetwork
from loss_functions import CrossEntropy, MeanSquaredError
from optimizer import StochasticGradientDescent, MomentumGradientDescent, NAG, Adagrad, RMSProp, Adam


class Configure:
      def __init__(self):
            self.hidden_layers = None
            self.input_size = None
            self.output_size = None
            self.activation_functions = None
            self.output_function = 'softmax'
            self.optimizer = None
            self.loss_function = None
            self.weight_init = None

      def show_blueprint(self):

            print("{hidden_layers =  // number of neuron in each layer e.g : [32, 128, 64] \ninput_size = //size of input eg:224\noutput_size = None //size of output eg:10\nactivation_functions = None // activation function to be used in each layer eg : relu/sigmoid/tanh \noutput_function = eg: softmax \noptimizer = //optimizer for parameter updation eg: Momentum Gradient Descent\nloss_function = //loss function to be used for gradient computation eg : Mean Squared Error / Cross Entropy \n}")
            
      def configure(self, script:dict):
            assert type(script['hidden_size']) == list 
            assert type(script['input_size']) == int
            assert type(script['output_size']) == int
            assert type(script['activation']) == str
            assert type(script['optimizer']) == str
            assert type(script['loss']) == str
            assert type(script['weight_init']) == str
            script['num_layers'] = len(script['hidden_size'])
            

            self.hidden_layers = script['hidden_size']
            self.input_size = script['input_size']
            self.output_size = script['output_size']
            self.activation_functions = script['activation']
            self.output_function = 'softmax'
            self.loss = {
                  'cross_entropy':CrossEntropy, 
                  'mean_squared_error':MeanSquaredError
            }
            self.loss_function = self.loss[script['loss'].lower()]
            self.weight_init = script['weight_init']
            self.optimizer_dict = {
                  "momentum": MomentumGradientDescent,
                  "sgd":StochasticGradientDescent,
                  "nag":NAG,
                  'adagrad':Adagrad,
                  'rmsprop': RMSProp,
                  'adam':Adam
            }
            self.optimizer = self.optimizer_dict[script['optimizer'].lower()]
            
            return NeuralNetwork(self.hidden_layers, self.input_size, self.output_size, self.activation_functions, self.output_function, self.weight_init), self.optimizer, self.loss_function

      

