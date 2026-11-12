import torch


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

class ScaleFFT(torch.nn.Module):
    def __init__(self, n_modes, D_in):
        super().__init__()
        self.n_modes = n_modes
        # self.ws = self._get_weights()
        self.ws = torch.nn.Parameter(self._get_weights())
        self.bias1 = torch.nn.Parameter(torch.Tensor(self.n_modes * D_in))
        self.bias2 = torch.nn.Parameter(torch.Tensor(self.n_modes * D_in))
        torch.nn.init.zeros_(self.bias1)
        torch.nn.init.zeros_(self.bias2)


    def _get_weights(self):
        return torch.arange(1, self.n_modes + 1, dtype=torch.float32).unsqueeze(0)

    def forward(self, x):
        ws = self.ws
        xx = torch.matmul(x[:, 0:1], ws)
        for i in range(1, x.shape[1]):
            xxx = torch.matmul(x[:, i:i + 1], ws)
            xx = torch.cat((xx, xxx), 1)
        x0 = torch.ones_like(x)
        out = torch.cat((x0, x, torch.sin(xx + self.bias1), torch.cos(xx + self.bias2)), 1)
        return out


class MultiLayerNet_FFT1(torch.nn.Module):
    def __init__(self, layers, n_modes=10, init=False):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(MultiLayerNet_FFT1, self).__init__()
        self.layers = list(layers)
        self.layers[1] = self.layers[0] * 2 * n_modes + 2 * self.layers[0]

        self.layer1 = ScaleFFT(n_modes, layers[0])
        self.linear_skip = torch.nn.Linear(layers[0], layers[-1])

        'activation function'
        self.activation = torch.nn.Tanh()

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
        y_skip = self.linear_skip(x)
        out = self.layer1(x)
        for i in range(len(self.layers)-3):
            z = self.linears[i+1](out)
            out = self.activation(z)
        out = self.linears[-1](out) + y_skip
        return out
