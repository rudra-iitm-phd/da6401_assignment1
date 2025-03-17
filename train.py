import numpy as np 
import matplotlib.pyplot as plt
import matplotlib
from data_loader import FashionMNISTData, MNISTData
from configure import Configure
import wandb
from trainer import Trainer
from score_metrics import Accuracy as accuracy
from argument_parser import parser
import sweep_configuration 
matplotlib.use("Agg") 
import time

wandb.login(key = "f7cc061a6cf1c6d4f2791a84e81d1d16ee8adc8b")

Datasets = {
      'mnist':MNISTData,
      'fashion_mnist':FashionMNISTData
}


def create_name(configuration:dict):
      l = [f'{k}-{v}' for k,v in configuration.items() if k not in ['input_size', 'output_size', 'wandb_entity', 'wandb_project', 'wandb_sweep', 'sweep_id', 'dataset']]
      return '_'.join(l)

def train():
      with wandb.init(entity = configuration_script['wandb_entity'],project = configuration_script['wandb_project'], name = create_name(configuration_script), config = configuration_script):

            
            sweep_config = wandb.config

            configuration_script.update(sweep_config)

            fig = data.display_collage((16, 14), wandb)
            
            # wandb.log({"Sample images from each class":fig})

            
            nn, optim, loss_fn = configuration.configure(configuration_script)

            nn.view_model_summary()

            trainer = Trainer(X_train, y_train, X_val, y_val, X_test, y_test, wandb)
            trainer.learn(nn=nn, optim=optim, loss_fn=loss_fn, lr=configuration_script['learning_rate'], batch_size=configuration_script['batch_size'], epochs = configuration_script['epochs'], acc_metrics=accuracy, loss = loss_fn, beta = configuration_script['beta'], forward=nn.forward, beta1 = configuration_script['beta1'], beta2 = configuration_script['beta2'], weight_decay = configuration_script['weight_decay'], eps = configuration_script['epsilon'])

            

            wandb.finish()

if __name__ == "__main__":

      args = parser.parse_args()

      data = Datasets[args.dataset.lower()](0.1)

      (X_train, y_train), (X_val, y_val), (X_test, y_test) = data.load_data(0.1)

      configuration = Configure()
      args.num_layers = len(args.hidden_size)

      print(args)

      configuration_script = args.__dict__
      if args.wandb_sweep:
            sweep_config = sweep_configuration.sweep_config
            if not args.sweep_id :
                  sweep_id = wandb.sweep(sweep_config, project=configuration_script['wandb_project'], entity=configuration_script['wandb_entity'])
            else:
                  sweep_id = args.sweep_id
            
            
            wandb.agent(sweep_id, function=train, count=20)
            # wandb.config.update({"sweep_name": create_name(wandb.config)})
            wandb.finish()
      else:
            train()