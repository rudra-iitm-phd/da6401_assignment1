import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import fashion_mnist
import matplotlib
matplotlib.use("Agg") 

class Data:

  def __init__(self, frac=1):

    (self.X_train, self.y_train), (self.X_val, self.y_val), (self.X_test, self.y_test) = self.load_data(frac) #loading the training, validation and test data along with labels

    self.classes = {
        0	: "T-shirt/top",
        1	: "Trouser",
        2	: "Pullover",
        3	: "Dress",
        4	: "Coat",
        5	: "Sandal",
        6	: "Shirt",
        7	: "Sneaker",
        8	: "Bag",
        9	: "Ankle boot"
    }

  def load_data(self, frac = 1):

    # using the keras fashion mnist to retrieve the train and test data
    (self.X_train, self.y_train) , (self.X_test, self.y_test) = fashion_mnist.load_data()

    #using the helper function to partition the training dataset into train and validation

    (self.X_train, self.y_train) , (self.X_val, self.y_val) = self.partition_data(self.X_train, self.y_train, frac)

    return (self.X_train, self.y_train), (self.X_val, self.y_val), (self.X_test, self.y_test)

  def partition_data(self, X : np.ndarray, y:np.ndarray, frac:float):

    # helper function to partition data into (1-frac)*100 % and frac * 100 %
    return (X[ : int(frac * X.shape[0]) ], y[ : int(frac * X.shape[0]) ]), (X[int( frac* X.shape[0]) : ], y[ int(frac * X.shape[0]) : ])

  def display_sample(self, k=10):

    # helper function to display any particular sample
    if not k:
      k = np.random.randint(0, self.X_train.shape[0])
    plt.imshow(self.X_train[k], cmap = 'grey')
    plt.title(self.classes[self.y_train[k]])

  def display_collage(self, figsize):

    # helper function to display samples from individual classes

    pop_list = list(range(10)) # a number list of integers from 0 --> 9
    fig, ax = plt.subplots(2, 5, figsize = figsize)
    i, j = 0, 0
    while len(pop_list) > 0:
      k = np.random.randint(0, len(self.X_train)) # we sample a number from the range of index available

      if self.y_train[k] in pop_list: # if this number is in the list we plot it and then pop it out

        ax[i, j].imshow(self.X_train[k], cmap = 'gray')
        ax[i, j].set_title(f'{self.classes[self.y_train[k]]}')

        pop_list.pop(pop_list.index(self.y_train[k]))

        # randomly placing images on the subplot

        i = (i+1) % 2  
        j = (j+1) % 5

    return fig