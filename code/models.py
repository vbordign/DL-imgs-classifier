import torch.nn as nn
from torch.nn import functional as F
import torch

class ConvNet(nn.Module):
    '''
    Creates a sequence of convolutional layers and RELU activations.

    Attributes
    ----------
    arch : list (int)
        a list of int values representing the number of channels
    kernel_conv : list (int)
        a list of int values representing the size of the convolution kernel

    Methods
    -------
    forward()
        runs a forward pass
    '''
    def __init__(self, arch, kernel_conv):
        super().__init__()
        self.conv_layers = nn.ModuleList([nn.Conv2d(arch[i], arch[i+1], kernel_size = kernel_conv[i]) for i in range(len(arch)-1)])

    def forward(self, x):
        for i, c in enumerate(self.conv_layers):
            x = F.relu(c(x))
        return x

class ConvNet2(nn.Module):
    '''
    Combines two parallel ConvNet modules with one common fully connected layer.

    Attributes
    ----------
    arch : list (int)
        a list of int values representing the number of channels
    kernel_conv : list (int)
        a list of int values representing the size of the convolution kernel
    fc_size: int
        size of the fully connected layer

    Methods
    -------
    forward()
        runs a forward pass
    '''
    def __init__(self, arch, kernel_conv, fc_size):
        super().__init__()
        self.cnn1 = ConvNet(arch, kernel_conv)
        self.cnn2 = ConvNet(arch, kernel_conv)
        self.fc_size = fc_size
        self.fc = nn.Linear(2*fc_size, 2)

    def forward(self, x):
        x1 = x[:,[0],:,:]
        x2 = x[:,[1],:,:]

        x1 = self.cnn1.forward(x1)
        x2 = self.cnn2.forward(x2)

        x = torch.cat((x1, x2), 1)

        x = x.view(-1, 2*self.fc_size)
        x = self.fc(x)
        return x



class ConvNet_WS(nn.Module):
    '''
    Combines two parallel ConvNet modules with weight sharing and
    one common fully connected layer.

    Attributes
    ----------
    arch : list (int)
        a list of int values representing the number of channels
    kernel_conv : list (int)
        a list of int values representing the size of the convolution kernel
    fc_size: int
        size of the fully connected layer

    Methods
    -------
    forward()
        runs a forward pass
    '''
    def __init__(self, arch, kernel_conv, fc_size):
        super().__init__()
        self.conv_layers = nn.ModuleList([nn.Conv2d(arch[i], arch[i+1], kernel_size = kernel_conv[i]) for i in range(len(arch)-1)])
        self.fc1 = nn.Linear(2 * fc_size , 2)
        self.fc_size = fc_size

    def forward(self, x):
        x1 = x[:,[0],:,:]
        x2 = x[:,[1],:,:]

        for i, c in enumerate(self.conv_layers):
            x1 = F.relu(c(x1))
        x1 = x1.view(-1, self.fc_size)

        for i, c in enumerate(self.conv_layers):
            x2 = F.relu(c(x2))
        x2 = x2.view(-1, self.fc_size)
        x = torch.cat((x1, x2), 1)
        x = self.fc1(x)
        return x

class ConvNet_WS_AL(nn.Module):
    '''
    Combines two parallel ConvNet modules with weight sharing and
    three fully connected layers, resulting in 3 outputs.

    Attributes
    ----------
    arch : list (int)
        a list of int values representing the number of channels
    kernel_conv : list (int)
        a list of int values representing the size of the convolution kernel
    fc_size: int
        size of the fully connected layer

    Methods
    -------
    forward()
        runs a forward pass
    '''
    def __init__(self, arch, kernel_conv, fc_size):
        super().__init__()
        self.conv_layers = nn.ModuleList([nn.Conv2d(arch[i], arch[i+1], kernel_size = kernel_conv[i]) for i in range(len(arch)-1)])
        self.fc1 = nn.Linear(2 * fc_size , 2)
        self.fc2 = nn.Linear(fc_size , 10)
        self.fc_size = fc_size
    def forward(self, x):
        x1 = x[:,[0],:,:]
        x2 = x[:,[1],:,:]

        for i, c in enumerate(self.conv_layers):
            x1 = F.relu(c(x1))
        x1 = x1.view(-1, self.fc_size)

        for i, c in enumerate(self.conv_layers):
            x2 = F.relu(c(x2))
        x2 = x2.view(-1, self.fc_size)
        x = torch.cat((x1, x2), 1)
        x = self.fc1(x)

        y1 = self.fc2(x1)
        y2 = self.fc2(x2)

        return x, y1, y2