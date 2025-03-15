# Neural Network for Fashion MNIST and MNIST image classification : from scratch
This project is a requirement for the completion of the course DA6401 Deep Learning taught by Prof Mitesh Khapra at IIT Madras. This readme file gives a brief overview about the project and if anyone's interested in replicating it, they can follow the instructions. 

The goal of the project is to build a neural network using python and taking help only from the following libraries : `numpy` , `pandas` and `matplotib`

Below is a short description of the project files 

| File name.py | Description       |
|--------------|-------------------|
|activation    |Contains the code for activation functions|
|argument_parser|Code for parsing the arguments provided by the user|
|configure|Code for using the parsed arguments and return a neural network, loss function and optimizer|
|data_loader|Code for loading the data from the choice (MNIST/Fashion MNIST) provided by the user|
|loss_functions|Code for computing loss and backpropagation|
|neural_network|Code for the neural network and inferencing|
|normalisation|Code for some basic normalisation processes|
|optimizer|Code for the various optimizers (sgd, momentum, nag, rmsprop and adam) and their update mechanisms|
|preprocess|Code for basic data processing|
|score_metrics|Code for computing accuracy|
|shared|Contains variables, methods and objects which can be shared amongst files|
|sweep_configuration|Contains the configuration for the wandb sweep|
|train|Contains the high level code for training th network|
|trainer|Code which integrates all the components to properly carry out the training and logging it to wandb|
|weight_init|Code for weight initialisation techniques (random/Xavier)|


## Steps to run it on your local machine
- Make sure you have python with version $\ge$`3.8.16` installed on your computer along with the following modules:
    - `numpy`
    - `pandas`
    - `matplotlib`
- Clone the git repository in your local machine by copying this piece of code
  ``` bash
  git clone https://github.com/rudra-iitm-phd/da6401_assignment1.git
  ```
- If you're on mac/linux you can proceed to terminal and change your directory to the folder where you've cloned the repository
- To just run the file with the default settings you can just use the following code
  ``` bash
  python train.py
  ```
- If you want to experiment out with various hidden layers, activation functions etc you can refer to the help by
  ``` bash
  python train.py -h
  ```
Following are the list of arguments which are compatible

|Name     |	Default Value	|Description|
|---------|-------------|----------|
|-wp, --wandb_project|	myprojectname	|Project name used to track experiments in Weights & Biases dashboard|
|-we, --wandb_entity|	myname	|Wandb Entity used to track experiments in the Weights & Biases dashboard.|
|-d, --dataset|	fashion_mnist	|choices: ["mnist", "fashion_mnist"]|
|-e, --epochs|	1	|Number of epochs to train neural network.|
|-b, --batch_size|	4	|Batch size used to train neural network.|
|-l, --loss|	cross_entropy	|choices: ["mean_squared_error", "cross_entropy"]|
|-o, --optimizer|	sgd	|choices: ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]|
|-lr, --learning_rate|	0.1	|Learning rate used to optimize model parameters|
|-m, --momentum|	0.5	|Momentum used by momentum and nag optimizers.|
|-beta, --beta|	0.5	|Beta used by rmsprop optimizer|
|-beta1, --beta1|	0.5	|Beta1 used by adam and nadam optimizers.|
|-beta2, --beta2|	0.5	|Beta2 used by adam and nadam optimizers.|
|-eps, --epsilon|	0.000001	|Epsilon used by optimizers.|
|-w_d, --weight_decay|	.0	|Weight decay used by optimizers.|
|-w_i, --weight_init|	random	|choices: ["random", "Xavier"]|
|-nhl, --num_layers|	1	|Number of hidden layers used in feedforward neural network.|
|-sz, --hidden_size|	4	|Number of hidden neurons in a feedforward layer.|
|-a, --activation|sigmoid	|choices: ["identity", "sigmoid", "tanh", "ReLU"]|
