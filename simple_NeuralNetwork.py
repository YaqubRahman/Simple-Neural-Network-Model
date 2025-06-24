import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd

url = 'https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv'
my_dataframe = pd.read_csv(url)

# Createing a Model Class that inherits nn.Module
class Model(nn.Module):
    # Input layer (4 features of the flower) 
    # Hidden Layer1 (number of neurons) 
    # H2 (n) 
    # output (3 classes of iris flowers)
    def __init__(self, in_features=4, h1=8, h2=9, out_features=3):
        super().__init__() # instantiate our nn.module
        self.fullyconnected1 = nn.Linear(in_features, h1)
        self.fullyconnected2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features) 


    # This function basically moves forward in the neural network
    def forwardmoving(self, x):
        # F refers to line 3
        # relu function is rectified linear unit 
        # relu - bascially do something and if the output is less than 0 just call it 0 otherwise its the output
        x = F.relu(self.fullyconnected1(x))
        x = F.relu(self.fullyconnected2(x))
        x = self.out(x)

        return x

# Pick a manual seed for randomisation
torch.manual_seed(41)
# Create an instance of model
model = Model()


my_dataframe['species'] = my_dataframe['species'].replace('setosa', 0.0)
my_dataframe['species'] = my_dataframe['species'].replace('versicolor', 1.0)
my_dataframe['species'] = my_dataframe['species'].replace('virginica', 2.0)

# Training, Testing and Spliting
# Setting x and y
x = my_dataframe.drop('species', axis=1)
y = my_dataframe['species']

x = x.values
y = y.values

print(my_dataframe)