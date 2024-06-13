import torch
import complextorch as cvtorch
from torch import nn

class ComplexValuedNN(nn.Module):
    """ This class defines a complex-valued neural network model. 
    It has two complex-valued convolutional layers, two complex-valued fully connected layers,
    with fully complex activations throughout the network. The final output layer is real-valued."""
    def __init__(self, n_conv_layers=2, conv_filters=[4,8], conv_kernel_size=[1,3], n_fc_layers=2, n_fc_units=128):
        super(ComplexValuedNN, self).__init__()
        self.conv_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()
        for i in range(n_conv_layers):
            if i == 0:
                self.conv_layers = nn.ModuleList([cvtorch.nn.CVConv1d(1, conv_filters[i], conv_kernel_size[i], padding="same")])
            else:
                self.conv_layers.append(cvtorch.nn.CVConv1d(conv_filters[i-1], conv_filters[i], conv_kernel_size[i], padding="same"))
            self.conv_layers.append(cvtorch.nn.CVCardiod())
            self.conv_layers.append(cvtorch.nn.CVBatchNorm1d(conv_filters[i]))
            # self.conv_layers.append(cvtorch.nn.CVDropout(0.25))

        if n_conv_layers > 0:
            self.flatten_size = conv_filters[-1] * 513  # Adjusted size to reflect conv2 output * sequence length
        else:
            self.flatten_size = 513
            
        for i in range(n_fc_layers):
            if i == 0:
                self.fc_layers = nn.ModuleList([cvtorch.nn.CVLinear(self.flatten_size, n_fc_units)])
            else:
                self.fc_layers.append(cvtorch.nn.CVLinear(n_fc_units, n_fc_units))
            self.fc_layers.append(cvtorch.nn.CVCardiod())
            # self.fc_layers.append(cvtorch.nn.CVBatchNorm1d(n_fc_units))
            # self.fc_layers.append(cvtorch.nn.CVDropout(0.25))
            
            
        # Output Layer (real-valued)

        self.output_layer = nn.Linear(n_fc_units * 2, 2)
        
    def forward(self, x):
        # Add channel dimension for convolution
        if len(self.conv_layers) > 0:
            x = x.unsqueeze(1)
        
        for layer in self.conv_layers:
            x = layer(x)
            
        # Flatten the tensor
        if len(self.conv_layers) > 0:
            x = x.view(x.size(0), -1)
        
        for layer in self.fc_layers:
            x = layer(x)
        
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
    def __init__(self, n_conv_layers=2, conv_filters=[8,16], conv_kernel_size=[1,3], n_fc_layers=2, n_fc_units=128):
        super(RealValuedNN, self).__init__()
        self.conv_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()
        for i in range(n_conv_layers):
            if i == 0:
                self.conv_layers = nn.ModuleList([nn.Conv1d(2, conv_filters[i], conv_kernel_size[i], padding="same")])
            else:
                self.conv_layers.append(nn.Conv1d(conv_filters[i-1], conv_filters[i], conv_kernel_size[i], padding="same"))
            self.conv_layers.append(nn.ReLU())
            self.conv_layers.append(nn.BatchNorm1d(conv_filters[i]))
            # self.conv_layers.append(nn.Dropout(0.25))

        if n_conv_layers > 0:
            self.flatten_size = conv_filters[-1] * 513
        else:
            self.flatten_size = 513 *2

        for i in range(n_fc_layers):
            if i == 0:
                self.fc_layers = nn.ModuleList([nn.Linear(self.flatten_size, n_fc_units)])
            else:
                self.fc_layers.append(nn.Linear(n_fc_units, n_fc_units))
            self.fc_layers.append(nn.ReLU())
            # self.fc_layers.append(nn.BatchNorm1d(n_fc_units))
            # self.fc_layers.append(nn.Dropout(0.25))
        
        # Output Layer (real-valued)
        self.output_layer = nn.Linear(n_fc_units, 2)
        
    def forward(self, x):
        # Combine real and imaginary parts as separate channels
        real = x.real
        imag = x.imag
        x = torch.stack((real, imag), dim=1)
        
        for layer in self.conv_layers:
            x = layer(x)
            
        # Flatten the tensor
        if len(self.conv_layers) > 0:
            x = x.view(x.size(0), -1)
        
        for layer in self.fc_layers:
            x = layer(x)
        
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
