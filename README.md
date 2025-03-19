# 🤖 Neural Network for Fashion MNIST and MNIST Image Classification (From Scratch)

This project is part of the **DA6401 Deep Learning** course taught by **Prof. Mitesh Khapra** at **IIT Madras**. The goal is to build a neural network using Python, relying only on the following libraries: 
- `numpy`
- `pandas`
- `matplotlib`

## 📄 Project Overview

This repository implements a neural network for classifying images from the **MNIST** and **Fashion MNIST** datasets. Below is a brief description of the project files:

| 📝 **File Name**          | 🔍 **Description**                                                  |
|--------------------------|--------------------------------------------------------------------|
| `activation.py`           | ⚡ Contains the code for activation functions.                      |
| `argument_parser.py`      | 🎯 Code for parsing user-provided arguments.                       |
| `configure.py`            | ⚙️ Configures the neural network, loss function, and optimizer.    |
| `data_loader.py`          | 📥 Loads data from MNIST or Fashion MNIST based on user choice.    |
| `loss_functions.py`       | 💀 Implements loss functions and backpropagation.                   |
| `neural_network.py`       | 🧠 Defines the neural network architecture and inference logic.    |
| `normalisation.py`        | 🧴 Contains basic data normalization processes.                    |
| `optimizer.py`            | 🛠️ Implements various optimizers (SGD, Momentum, NAG, RMSProp, Adam). |
| `preprocess.py`           | 🔄 Contains basic data preprocessing steps.                        |
| `score_metrics.py`        | 📊 Computes accuracy and other performance metrics.                |
| `shared.py`               | 🔗 Shared variables, methods, and objects used across multiple files. |
| `sweep_configuration.py`  | 🧹 Configuration for Weights & Biases sweep.                       |
| `train.py`                | 🚀 High-level code for training the neural network.               |
| `trainer.py`              | 🤖 Integrates components to manage training and logging to Weights & Biases. |
| `weight_init.py`          | 🏋️‍♂️ Implements weight initialization techniques (Random, Xavier). |


## ⚙️ Getting Started

Follow the steps below to run the project locally.

### 🛠️ Prerequisites

Ensure you have Python 3.8.16 or higher installed along with the following packages:
- `numpy`
- `pandas`
- `matplotlib`

### 🚀 Clone the Repository

Clone the project to your local machine by running the following command:

```bash
git clone https://github.com/rudra-iitm-phd/da6401_assignment1.git
```
### 🖥️ Run the Model
Navigate to the directory where the repository was cloned. To run the model with default settings, execute:
Following are the list of arguments which are compatible

```bash
python train.py
```
### 🔧 Customize the Model
To customize parameters like hidden layers, activation functions, etc., run with the arguemnts with the respective values provided by a space
for eg :
```bash
python train.py -b 1024 -o adam -w_d 1e-2 -e 500 -beta1 0.9 -beta2 0.9999  -w_i xavier -lr 1e-2 -a tanh
```
Following is a screen shot provided post running this command

<img width="1440" alt="Screenshot 2025-03-17 at 11 07 34 PM" src="https://github.com/user-attachments/assets/317ed21a-daa7-4303-b008-f0fb1e180858" />



If you need to check out the list of the parameters available run:

``` bash
python train.py -h
```
### 🌟 List of Arguments 

| 🔧 **Name**              | 🔄 **Default Value**   | 📜 **Description**                                                       |
|--------------------------|------------------------|-------------------------------------------------------------------------|
| `-wp`, `--wandb_project`    | `myprojectname`        | 🧑‍💻 Project name for tracking experiments in the Weights & Biases dashboard. |
| `-we`, `--wandb_entity`     | `myname`               | 🧑‍💻 Weights & Biases entity for tracking experiments.                 |
| `-d`, `--dataset`           | `fashion_mnist`        | 📊 Choices: `["mnist", "fashion_mnist"]`                                |
| `-e`, `--epochs`            | `500`                    | ⏳ Number of epochs to train the neural network.                        |
| `-b`, `--batch_size`        | `1024`                    | 📦 Batch size used to train the neural network.                         |
| `-l`, `--loss`              | `cross_entropy`        | 💀 Choices: `["mean_squared_error", "cross_entropy"]`                   |
| `-o`, `--optimizer`         | `adam`                  | ⚙️ Choices: `["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]`    |
| `-lr`, `--learning_rate`    | `1e-2`                  | 🚀 Learning rate used to optimize model parameters.                     |
| `-m`, `--momentum`          | `0.5`                  | 🌀 Momentum used by momentum and NAG optimizers.                         |
| `-beta`, `--beta`           | `0.5`                  | 🔢 Beta used by RMSProp optimizer.                                      |
| `-beta1`, `--beta1`         | `0.9`                  | 🔢 Beta1 used by Adam and Nadam optimizers.                             |
| `-beta2`, `--beta2`         | `0.9999`                  | 🔢 Beta2 used by Adam and Nadam optimizers.                             |
| `-eps`, `--epsilon`         | `0.000001`             | 🔢 Epsilon used by optimizers to prevent division by zero.              |
| `-w_d`, `--weight_decay`    | `1e-2`                  | ⚖️ Weight decay used by optimizers to regularize the model.             |
| `-w_i`, `--weight_init`     | `xavier`               | 🎲 Choices: `["random", "Xavier"]`                                     |
| `-nhl`, `--num_layers`      | `1`                    | 🧠 Number of hidden layers in the neural network.                       |
| `-sz`, `--hidden_size`      | `784`                    | 🔲 Number of neurons in each hidden layer.                              |
| `-a`, `--activation`        | `relu`              | 🌟 Choices: `["identity", "sigmoid", "tanh", "ReLU"]`                   |
| `--wandb_sweep`   | `false`       | 🧑‍💻 To generate a wandb sweep with the sweep configuration |
| `--sweep_id`   | `null`      | 🧑‍💻 Run a sweep agent with a specific sweep id |
### 💬 Contact
For any questions or issues, feel free to open an issue or contact me via GitHub.
