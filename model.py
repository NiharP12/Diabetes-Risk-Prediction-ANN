import torch
import torch.nn as nn

class DiabetesANN(nn.Module):
    def __init__(self, input_size):
        super(DiabetesANN, self).__init__()
        # Hidden Layer 1: 16 neurons
        self.fc1 = nn.Linear(input_size, 16)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        
        # Hidden Layer 2: 8 neurons
        self.fc2 = nn.Linear(16, 8)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.1)
        
        # Output Layer: 1 neuron (binary classification)
        self.fc3 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x
