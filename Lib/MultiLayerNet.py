import torch

from Lib.ImportLibrary import *


class MultiLayerNet(torch.nn.Module):
    def __init__(self, layers, init=False):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(MultiLayerNet, self).__init__()
        self.layers = layers

        'activation function'
        self.activation = torch.nn.Tanh()
        # self.activation = torch.nn.Tanh()

        'Initialise neural network as a list using nn.Modulelist'
        self.linears = torch.nn.ModuleList([torch.nn.Linear(self.layers[i], self.layers[i + 1]) for i in range(len(self.layers) - 1)])

        if init == True:
            'Xavier Normal Initialization'
            for i in range(len(layers) - 1):
                torch.nn.init.xavier_normal_(self.linears[i].weight.data)
                # set biases to zero
                torch.nn.init.zeros_(self.linears[i].bias.data)
            print('Constructor: Init Network!')

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        out = x
        for i in range(len(self.layers)-2):
            z = self.linears[i](out)
            out = self.activation(z)
        out = self.linears[-1](out)
        return out
