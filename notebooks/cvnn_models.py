import torch
import complextorch as cvtorch
from torch import nn

class ComplexValuedNN(nn.Module):
    """ This class defines a complex-valued neural network model. 
    It has two complex-valued convolutional layers, two complex-valued fully connected layers,
    with fully complex activations throughout the network. The final output layer is real-valued."""
    def __init__(self):
        super(ComplexValuedNN, self).__init__()

        # Complex Conv Layer 1
        self.conv1 = cvtorch.nn.CVConv1d(1, 4, 1, padding="same")
        self.activation1 = cvtorch.nn.CVSigLog()
        self.bn1 = cvtorch.nn.CVBatchNorm1d(4)
        # self.dropout1 = cvtorch.nn.CVDropout(0.25)

        # Complex Conv Layer 2
        self.conv2 = cvtorch.nn.CVConv1d(4, 8, 3, padding="same")
        self.activation2 = cvtorch.nn.CVSigLog()
        self.bn2 = cvtorch.nn.CVBatchNorm1d(8)
        # self.dropout2 = cvtorch.nn.CVDropout(0.25)

        # Fully connected Layer 1
        self.fc1 = cvtorch.nn.CVLinear(8 * 513, 128)
        self.activation3 = cvtorch.nn.CVSigLog()
        # self.bn3 = cvtorch.nn.CVBatchNorm1d(1024)
        # self.dropout3 = cvtorch.nn.CVDropout(0.25)

        # Fully connected Layer 2
        # self.fc2 = cvtorch.nn.CVLinear(1024, 1024)
        # self.activation4 = cvtorch.nn.zReLU()
        # self.bn4 = cvtorch.nn.CVBatchNorm1d(1024)
        # self.dropout4 = cvtorch.nn.CVDropout(0.25)
        # Fully connected Layer 2
        self.fc3 = cvtorch.nn.CVLinear(128, 32)
        self.activation5 = cvtorch.nn.CVSigLog()
        
        # self.bn5 = cvtorch.nn.CVBatchNorm1d(32)
        # self.dropout5 = cvtorch.nn.CVDropout(0.25)
        
        # Output Layer (real-valued)
        self.output_layer = nn.Linear(32 * 2, 2)

    def forward(self, x):
        # Add channel dimension for convolution
        x = x.unsqueeze(1)
        # x = x.reshape(x.shape[1], 1, -1)

        # Conv Layer 1
        x = self.conv1(x)
        x = self.activation1(x)
        x = self.bn1(x)
        # x = self.dropout1(x)

        # # # Conv Layer 2
        x = self.conv2(x)
        x = self.activation2(x)
        x = self.bn2(x)
        # x = self.dropout2(x)

        # Flatten the tensor
        x = x.view(x.size(0), -1)

        # Fully Connected Layer 1
        x = self.fc1(x)
        x = self.activation3(x)
        # x = self.bn3(x)
        # x = self.dropout3(x)

        # Fully Connected Layer 2
        # x = self.fc2(x)
        # x = self.activation4(x)
        # x = self.bn4(x)
        # x = self.dropout4(x)
        x = self.fc3(x)
        x = self.activation5(x)
        # x = self.bn5(x)
        # x = self.dropout5(x)
        # Transform complex valued output for the final real-valued layer
        real_x = torch.cat((x.real, x.imag), dim=1)  # Merge real and imag parts

        # Output Layer
        out = self.output_layer(real_x)

        return out


class RealValuedNN(nn.Module):
    """ This class defines a real-valued neural network model.
    It has two real-valued convolutional layers, two real-valued fully connected layers,
    with real-valued activations throughout the network. The final output layer is real-valued.
    
    The real and imaginary parts of the data are combined as separate channels."""
    def __init__(self):
        super(RealValuedNN, self).__init__()

        # Conv Layer 1
        self.conv1 = nn.Conv1d(2, 8, kernel_size=1, padding='same')

        # Activation 1
        self.activation1 = nn.ReLU()
        
        # BatchNorm 1
        self.bn1 = nn.BatchNorm1d(8)

        # Conv Layer 2
        self.conv2 = nn.Conv1d(8, 16, kernel_size=3, padding='same')

        # Activation 2
        self.activation2 = nn.ReLU()
        
        # BatchNorm 2
        self.bn2 = nn.BatchNorm1d(16)

        # Given input size is (513), Convolution does not alter the dimension with padding='same'
        # Therefore, the size is the same after convolutions (513).
        # Flatten the tensor
        self.flatten_size = 16 * 513  # Adjusted size to reflect conv2 output * sequence length

        # Fully connected Layer 1
        self.fc1 = nn.Linear(self.flatten_size, 128)

        # Activation 3
        self.activation3 = nn.ReLU()

        # Fully connected Layer 2
        self.fc2 = nn.Linear(128, 32)

        # Activation 4
        self.activation4 = nn.ReLU()
        
        # Output Layer (real-valued)
        self.output_layer = nn.Linear(32, 2)

    def forward(self, x):
        # Combine real and imaginary parts as separate channels
        real = x.real
        imag = x.imag
        x = torch.stack((real, imag), dim=1)

        # Conv Layer 1
        x = self.conv1(x)
        x = self.activation1(x)
        x = self.bn1(x)

        # Conv Layer 2
        x = self.conv2(x)
        x = self.activation2(x)
        x = self.bn2(x)

        # Flatten the tensor
        x = x.view(x.size(0), -1)
        
        # Fully Connected Layer 1
        x = self.fc1(x)
        x = self.activation3(x)

        # Fully Connected Layer 2
        x = self.fc2(x)
        x = self.activation4(x)

        # Output Layer
        out = self.output_layer(x)

        return out
    
class RealValuedNNtd(nn.Module):
    """ This class defines a real-valued neural network model.
    It has two real-valued convolutional layers, two real-valued fully connected layers,
    with real-valued activations throughout the network. The final output layer is real-valued.
    
    This model is for time-domain data."""
    def __init__(self):
        super(RealValuedNNtd, self).__init__()

        # Conv Layer 1
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding='same')
        self.maxpool1 = nn.MaxPool1d(2)

        # Activation 1
        self.activation1 = nn.ReLU()
        
        # BatchNorm 1
        self.bn1 = nn.BatchNorm1d(64)

        # Conv Layer 2
        self.conv2 = nn.Conv1d(64, 128, kernel_size=6, padding='same')
        self.maxpool2 = nn.MaxPool1d(2)

        # Activation 2
        self.activation2 = nn.ReLU()
        
        # BatchNorm 2
        self.bn2 = nn.BatchNorm1d(128)

        # Flatten the tensor
        self.lstm = nn.LSTM(input_size=128, hidden_size=32, num_layers=3, batch_first=True)
        self.flatten_size = 128 * 256  # Adjusted size to reflect conv2 output * sequence length
        # Fully connected Layer 1
        # self.fc1 = nn.Linear(self.flatten_size, 256)

        # Activation 3
        # self.activation3 = nn.ReLU()

        # Fully connected Layer 2
        # self.fc2 = nn.Linear(64, 32)

        # Activation 4
        # self.activation4 = nn.ReLU()
        
        # Output Layer (real-valued)
        self.output_layer = nn.Linear(32, 2)

    def forward(self, x):
        # Combine real and imaginary parts as separate channels
        x = x.unsqueeze(1)

        # Conv Layer 1
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.activation1(x)
        x = self.bn1(x)

        # Conv Layer 2
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.activation2(x)
        x = self.bn2(x)
        x = x.permute(0, 2, 1)
        x = self.lstm(x)
        # Flatten the tensor
        # x = x.view(x.size(0), -1)
        
        # Fully Connected Layer 1
        # x = self.fc1(x)
        # x = self.activation3(x)

        # Fully Connected Layer 2
        # x = self.fc2(x)
        # x = self.activation4(x)

        # Output Layer
        out = self.output_layer(x[:, -1, :])

        return out
