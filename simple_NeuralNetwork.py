import torch
import torch.nn as nn
import torch.nn.functional as F

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