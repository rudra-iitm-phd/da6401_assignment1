import numpy as np 
import matplotlib.pyplot as plt
from data_loader import Data
from configure import Configure
# import wandb
from trainer import Trainer
from score_metrics import Accuracy as accuracy
from argument_parser import parser

# wandb.login(key = "f7cc061a6cf1c6d4f2791a84e81d1d16ee8adc8b")


data = Data(0.1)
(X_train, y_train), (X_val, y_val), (X_test, y_test) = data.load_data(0.1)

configuration = Configure()

args = parser.parse_args()

configuration_script = args.__dict__

def create_name(configuration:dict):
      l = [f'{k}-{v}' for k,v in configuration.items() if k not in ['input_size', 'output_size']]
      return '_'.join(l)


# wandb.init(project="da24d008-assignment1", name = create_name(configuration_script), config = configuration_script)



fig = data.display_collage((16, 14))
# wandb.log({"Sample images from each class":fig})


nn, optim, loss_fn = configuration.configure(configuration_script)

nn.view_model_summary()

trainer = Trainer(X_train, y_train, X_val, y_val, None)
trainer.learn(nn=nn, optim=optim, loss_fn=loss_fn, lr=configuration_script['learning_rate'], batch_size=configuration_script['batch_size'], epochs = configuration_script['epochs'], acc_metrics=accuracy, loss = loss_fn, beta = configuration_script['beta'], forward=nn.forward)