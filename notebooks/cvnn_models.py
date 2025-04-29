import torch
import complextorch.nn as cvnn
from torch import nn


class SimpleComplexBN1d(nn.Module):
    """
    Works for (B, C) *or* (B, C, L) complex.
    Internally runs real BatchNorm on the concatenated
    [real, imag] channels.
    """

    def __init__(self, num_features, **bn_kw):
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(num_features * 2, **bn_kw)

    def forward(self, z):                         # complex
        orig_shape = z.shape                      # save for later
        if z.dim() == 2:                          # (B, C)
            z = z.unsqueeze(-1)                   # → (B, C, 1)

        B, C, L = z.shape
        ri = torch.view_as_real(z)                # (B, C, L, 2)
        ri = ri.permute(0, 3, 1, 2)               # (B, 2, C, L)
        ri = ri.reshape(B, 2 * C, L)              # (B, 2C, L)  real
        ri = self.bn(ri)
        ri = ri.view(B, 2, C, L).permute(0, 2, 3, 1).contiguous()
        z = torch.view_as_complex(ri)            # (B, C, L)

        return z.squeeze(-1) if len(orig_shape) == 2 else z

class ComplexValuedNN(nn.Module):
    """
    Two complex 1-D conv blocks → complex global-avg-pool →
    two complex FC blocks → real regression head (2 outputs).
    """

    def __init__(self,
                 n_conv_layers=2,
                 conv_filters=[4, 8],
                 conv_kernel_size=[1, 3],
                 n_fc_layers=2,
                 n_fc_units=128,
                 dropout=None,
                 pool_size: int = 1):     # 1 = true global average pooling
        super().__init__()

        # ---------- Convolutional trunk ----------
        self.conv_layers = nn.ModuleList()
        in_ch = 1
        for i in range(n_conv_layers):
            self.conv_layers.append(
                cvnn.Conv1d(in_ch,
                              conv_filters[i],
                              conv_kernel_size[i],
                              padding="same")
            )
            self.conv_layers.append(cvnn.CReLU())
            self.conv_layers.append(SimpleComplexBN1d(conv_filters[i]))
            if dropout:
                self.conv_layers.append(cvnn.Dropout(dropout))
            in_ch = conv_filters[i]

        # ---------- Complex global / adaptive pooling ----------
        self.gap = cvnn.AdaptiveAvgPool1d(
            pool_size)   # keeps C×pool_size features

        # ---------- Fully–connected complex head ----------
        fc_in = conv_filters[-1] * pool_size           # ← correct input size
        self.fc_layers = nn.ModuleList()
        for i in range(n_fc_layers):
            out_units = n_fc_units
            self.fc_layers.append(
                cvnn.Linear(fc_in if i == 0 else n_fc_units, out_units)
            )
            self.fc_layers.append(cvnn.CReLU())
            # if i != n_fc_layers - 1:
            #     self.fc_layers.append(SimpleComplexBN1d(out_units))
            if dropout:
                self.fc_layers.append(cvnn.Dropout(dropout))

        # ---------- Real-valued regression head ----------
        self.output_layer = nn.Linear(n_fc_units * 2, 5)

    # ----------------------------------------------------
    def forward(self, x):
        # x shape: (B, 513)  →  (B, 1, 513)
        if x.dim() == 2:              # (B , 513)
            x = x.unsqueeze(1)        # → (B , 1 , 513) ✔
        for layer in self.conv_layers:
            x = layer(x)                 # (B, C, 513)

        x = self.gap(x)                  # (B, C, pool_size)
        x = x.flatten(1, 2)

        for layer in self.fc_layers:
            x = layer(x)

        # convert complex to real before final layer
        x = torch.cat((x.real, x.imag), dim=1)   # (B, 2·n_fc_units)
        return self.output_layer(x)

class RealValuedNN(nn.Module):
    """ This class defines a real-valued neural network model.
    It has two real-valued convolutional layers, two real-valued fully connected layers,
    with real-valued activations throughout the network. The final output layer is real-valued.

    The real and imaginary parts of the data are combined as separate channels."""
    def __init__(self, n_conv_layers=2, conv_filters=[8,16], conv_kernel_size=[1,3], n_fc_layers=2, n_fc_units=128, dropout=None):
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
            if dropout:
                self.conv_layers.append(nn.Dropout(dropout))

        if n_conv_layers > 0:
            self.flatten_size = conv_filters[-1] * 513
        else:
            self.flatten_size = 513

        for i in range(n_fc_layers):
            if i == 0:
                self.fc_layers = nn.ModuleList([nn.Linear(self.flatten_size, n_fc_units)])
            else:
                self.fc_layers.append(nn.Linear(n_fc_units, n_fc_units))
            self.fc_layers.append(nn.ReLU())
            if i != n_fc_layers - 1:
                self.fc_layers.append(nn.BatchNorm1d(n_fc_units))
            if dropout:
                self.fc_layers.append(nn.Dropout(dropout))

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
