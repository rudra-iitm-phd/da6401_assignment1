from neural_network import NeuralNetwork
from loss_functions import CrossEntropy
from optimizer import GradientDescent, MomentumGradientDescent


class Configure:
      def __init__(self):
            self.hidden_layers = None
            self.input_size = None
            self.output_size = None
            self.activation_functions = None
            self.output_function = 'softmax'
            self.optimizer = None
            self.loss_function = None

      def show_blueprint(self):

            print("{hidden_layers =  // number of neuron in each layer e.g : [32, 128, 64] \ninput_size = //size of input eg:224\noutput_size = None //size of output eg:10\nactivation_functions = None // activation function to be used in each layer eg : relu/sigmoid/tanh \noutput_function = eg: softmax \noptimizer = //optimizer for parameter updation eg: Momentum Gradient Descent\nloss_function = //loss function to be used for gradient computation eg : Mean Squared Error / Cross Entropy \n}")
            
      def configure(self, script:dict):
            assert type(script['hidden_layers']) == list 
            assert type(script['input_size']) == int
            assert type(script['output_size']) == int
            assert type(script['activation_functions']) == list 
            assert type(script['optimizer']) == str
            assert type(script['loss_function']) == str

            self.hidden_layers = script['hidden_layers']
            self.input_size = script['input_size']
            self.output_size = script['output_size']
            self.activation_functions = script['activation_functions']
            self.output_function = 'softmax'
            self.optimizer = MomentumGradientDescent if script['optimizer'] == 'momentum' else GradientDescent
            self.loss_function = CrossEntropy if script['loss_function'] == 'cross entropy' else None

            return NeuralNetwork(self.hidden_layers, self.input_size, self.output_size, self.activation_functions, self.output_function), self.optimizer, self.loss_function

      

