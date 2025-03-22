# -*- coding = utf-8 -*-
# @time:2025/2/20 14:16
# Author:Zhihao.Xiong
# @File:myFCNGrad.py
# @Software:PyCharm
from Lib.ImportLibrary import *

class Conv2dDerivative(torch.nn.Module):
    def __init__(self, DerFilter, resol, kernel_size=3, name=''):
        super(Conv2dDerivative, self).__init__()

        self.resol = resol  # constant in the finite difference
        self.name = name
        self.input_channels = 1
        self.output_channels = 1
        self.kernel_size = kernel_size

        self.padding = int((kernel_size - 1) / 2)
        self.filter = torch.nn.Conv2d(self.input_channels, self.output_channels, self.kernel_size,
            1, padding=0, bias=False)

        # Fixed gradient operator
        self.filter.weight = torch.nn.Parameter(torch.FloatTensor(DerFilter), requires_grad=False)

    def forward(self, input):
        derivative = self.filter(input)
        return derivative / self.resol

class Conv3dDerivative(torch.nn.Module):
    def __init__(self, DerFilter, resol, kernel_size=3, name=''):
        super(Conv3dDerivative, self).__init__()

        self.resol = resol  # constant in the finite difference
        self.name = name
        self.input_channels = 1
        self.output_channels = 1
        self.kernel_size = kernel_size

        self.padding = int((kernel_size - 1) / 2)
        self.filter = torch.nn.Conv3d(self.input_channels, self.output_channels, self.kernel_size,
                                      (1,1,1), padding=0, bias=False)

        # Fixed gradient operator
        self.filter.weight = torch.nn.Parameter(DerFilter, requires_grad=False)

    def forward(self, input):
        derivative = self.filter(input)
        return derivative / self.resol


class Derivative(torch.nn.Module):
    ''' Loss generator for physics loss '''

    def __init__(self, dx, dy, dz=None, KernelSize=5, dim=2):
        super(Derivative, self).__init__()

        ''' Construct the derivatives, X = Width, Y = Height '''
        if dim==2:
            if KernelSize == 3:
                partial_x = [[[[0, 0, 0],
                               [-1/2, 0, 1/2],
                               [0, 0, 0]]]]

                partial_y = [[[[0, -1/2, 0],
                               [0, 0, 0],
                               [0, 1/2, 0]]]]

                # lapl_op = [[[[0, dy, 0],
                #              [dx, -4*dx*dy, dx],
                #              [0, dy, 0]]]]

                lapl_op = [[[[0, 1, 0],
                             [1, -4, 1],
                             [0, 1, 0]]]]

            elif KernelSize == 5:
                partial_x = [[[[0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0],
                               [1 / 12, -8 / 12, 0, 8 / 12, -1 / 12],
                               [0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0]]]]

                partial_y = [[[[0, 0, 1 / 12, 0, 0],
                               [0, 0, -8 / 12, 0, 0],
                               [0, 0, 0, 0, 0],
                               [0, 0, 8 / 12, 0, 0],
                               [0, 0, -1 / 12, 0, 0]]]]

                lapl_op = [[[[0, 0, -1 / 12, 0, 0],
                             [0, 0, 4 / 3, 0, 0],
                             [-1 / 12, 4 / 3, -5, 4 / 3, -1 / 12],
                             [0, 0, 4 / 3, 0, 0],
                             [0, 0, -1 / 12, 0, 0]]]]
            elif KernelSize == 7:
                partial_x = [[[[0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0],
                             [-1/60, 9/60, -45/60, 0, 45/60, -9/60, 1/60],
                             [0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0]]]]

                partial_y = [[[[0, 0, 0, -1/60, 0, 0, 0],
                             [0, 0, 0, 9/60, 0, 0, 0],
                             [0, 0, 0, -45/60, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 45/60, 0, 0, 0],
                             [0, 0, 0, -9/60, 0, 0, 0],
                             [0, 0, 0, 1/60, 0, 0, 0]]]]

                lapl_op = [[[[0, 0, -1 / 12, 0, 0],
                             [0, 0, 4 / 3, 0, 0],
                             [-1 / 12, 4 / 3, -5, 4 / 3, -1 / 12],
                             [0, 0, 4 / 3, 0, 0],
                             [0, 0, -1 / 12, 0, 0]]]]

            elif KernelSize == 9:
                partial_x = [[[[0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [1 / 840, -32 / 840, 168 / 840, -672 / 840, 0, 672 / 840, -168 / 840, 32 / 840, -1 / 840],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0]]]]

                partial_y = [[[[0, 0, 0, 0, 1 / 840, 0, 0, 0, 0],
                               [0, 0, 0, 0, -32 / 840, 0, 0, 0, 0],
                               [0, 0, 0, 0, 168 / 840, 0, 0, 0, 0],
                               [0, 0, 0, 0, -672 / 840, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 672 / 840, 0, 0, 0, 0],
                               [0, 0, 0, 0, -168 / 840, 0, 0, 0, 0],
                               [0, 0, 0, 0, 32 / 840, 0, 0, 0, 0],
                               [0, 0, 0, 0, -1 / 840, 0, 0, 0, 0]]]]

                lapl_op = [[[[0, 0, -1 / 12, 0, 0],
                             [0, 0, 4 / 3, 0, 0],
                             [-1 / 12, 4 / 3, -5, 4 / 3, -1 / 12],
                             [0, 0, 4 / 3, 0, 0],
                             [0, 0, -1 / 12, 0, 0]]]]

            else:
                raise NotImplementedError

            self.dx = Conv2dDerivative(
                DerFilter=partial_x,
                resol=(dx * 1),
                kernel_size=KernelSize,
                name='dx_operator').to(dev)

            self.dy = Conv2dDerivative(
                DerFilter=partial_y,
                resol=(dy * 1),
                kernel_size=KernelSize,
                name='dy_operator').to(dev)

            self.laplace = Conv2dDerivative(
                DerFilter = lapl_op,
                resol = (dx*dy),
                kernel_size = 5,
                name = 'laplace_operator').to(dev)

        if dim==3:
            if KernelSize == 3:
                partial_x = torch.tensor([
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                    [[0, 0, 0], [-0.5, 0, 0.5], [0, 0, 0]],
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]).float()

                partial_y = torch.tensor([
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                    [[0, -0.5, 0], [0, 0, 0], [0, 0.5, 0]],
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]).float()

                partial_z = torch.tensor([
                    [[0, 0, 0], [0, -0.5, 0], [0, 0, 0]],
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                    [[0, 0, 0], [0, 0.5, 0], [0, 0, 0]]]).float()

                # lapl_op = [[[[0, dy, 0],
                #              [dx, -4*dx*dy, dx],
                #              [0, dy, 0]]]]

                # lapl_op = [[[[0, 1, 0],
                #              [1, -4, 1],
                #              [0, 1, 0]]]]

            elif KernelSize == 5:
                partial_z = torch.tensor([
                    [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                    [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                    [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1 / 12, -8 / 12, 0, 8 / 12, -1 / 12], [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0]],
                    [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                    [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]]).float()

                partial_x = torch.tensor([
                    [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                    [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                    [[0, 0, 1 / 12, 0, 0], [0, 0, -8 / 12, 0, 0], [0, 0, 0, 0, 0], [0, 0, 8 / 12, 0, 0],[0, 0, -1 / 12, 0, 0]],
                    [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                    [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]]).float()

                partial_y = torch.tensor([
                    [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1 / 12, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                    [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, -8 / 12, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                    [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],[0, 0, 0, 0, 0]],
                    [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 8 / 12, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                    [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, -1 / 12, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]]).float()

                # lapl_op = [[[[0, 0, -1 / 12, 0, 0],
                #              [0, 0, 4 / 3, 0, 0],
                #              [-1 / 12, 4 / 3, -5, 4 / 3, -1 / 12],
                #              [0, 0, 4 / 3, 0, 0],
                #              [0, 0, -1 / 12, 0, 0]]]]

            else:
                raise NotImplementedError

            partial_x = partial_x.unsqueeze(0).unsqueeze(0)
            partial_y = partial_y.unsqueeze(0).unsqueeze(0)
            partial_z = partial_z.unsqueeze(0).unsqueeze(0)

            self.dx = Conv3dDerivative(
                DerFilter=partial_x,
                resol=(dx * 1),
                kernel_size=KernelSize,
                name='dx_operator').to(dev)

            self.dy = Conv3dDerivative(
                DerFilter=partial_y,
                resol=(dy * 1),
                kernel_size=KernelSize,
                name='dy_operator').to(dev)

            self.dz = Conv3dDerivative(
                DerFilter=partial_z,
                resol=(dz * 1),
                kernel_size=KernelSize,
                name='dz_operator').to(dev)