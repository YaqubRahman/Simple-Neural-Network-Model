import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

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
        # The different layers
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

# Converting them into numpy arrays
x = x.values
y = y.values

# Train Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=41)

# Convert x features to float tensors
x_train = torch.FloatTensor(x_train)
x_test = torch.FloatTensor(x_test)

# Convert y labels to tensors long
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# Set the criterion of model to measure the error, how far off the predictions are from the data
criterion = nn.CrossEntropyLoss()
# Choose Optimiser (Adam optimiser), lr = learning rate
# learning rate -> (if error doesnt go down after a bunch of iterations (epochs) probably want to lower the learning rate)
optimiser = torch.optim.Adam(model.parameters(), lr=0.01)



# Training our model
# Epochs (one run thru all the training data in our network)
epochs = 100
losses = []
for i in range(epochs):
    # Go forward in the network and get a prediction
    # Get predicted results
    y_prediction = model.forwardmoving(x_train)

    # Measure the loss/error, will be high at first
    # Predicted values vs the y_train
    loss = criterion(y_prediction, y_train)

    # Keep track of our losses
    losses.append(loss.detach().numpy())

    # Print every 10 epoch
    if i % 10 == 0:
        print(f'Epoch: {i} and loss: {loss}')

    # Do some back propagation
    # Back propagation: take the error rate of forward propagation and fee it back
    # thru the network to fine tune the weights
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()



# Graphing it out
#plt.plot(range(epochs), losses)
#plt.ylabel("loss/error")
#plt.xlabel('Epoch')
#plt.show()

# print(model.parameters)

# Evaluate Model on Test Data Set (validate model on test set)

# Basically turn off back propagation
with torch.no_grad():
    y_eval = model.forwardmoving(x_test)
    # Find the loss or error 
    loss = criterion(y_eval, y_test) 

#print(loss)

correct = 0
with torch.no_grad():
    for i, data in enumerate(x_test):
        y_val = model.forwardmoving(data)

        # Will tell us what type of flower class our network thinks it is
        print(f'{i+1}.) {str(y_val)} \t {y_test[i]}')

        # Correct or not
        if y_val.argmax().item() == y_test[i]:
            correct += 1

    print(f'We got {correct} correct! ')
