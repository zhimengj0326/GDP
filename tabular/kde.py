import torch
from torch import nn
from math import pi, sqrt
import numpy as np

class kde_fair:
    """
    A Gaussian KDE implemented in pytorch for the gradients to flow in pytorch optimization.
    Keep in mind that KDE are not scaling well with the number of dimensions and this implementation is not really
    optimized...
    """
    def __init__(self, x_test):
        # self.train_x = x_train
        # self.train_y = y_train
        self.x_test = x_test
    
    def forward(self, y_train, x_train, device_gpu):
        n = x_train.size()[0]
        # print(f'n={n}')
        d = 1
        bandwidth = torch.tensor((n * (d + 2) / 4.) ** (-1. / (d + 4))).to(device_gpu)

        y_hat = self.kde_regression(bandwidth, x_train, y_train)
        y_mean = torch.mean(y_train)
        pdf_values = self.pdf(bandwidth, x_train)

        DP = torch.sum(torch.abs(y_hat-y_mean) * pdf_values) / torch.sum(pdf_values)
        return DP

    def kde_regression(self, bandwidth, x_train, y_train):
        n = x_train.size()[0]
        X_repeat = self.x_test.repeat_interleave(n).reshape((-1, n))
        attention_weights = nn.functional.softmax(-(X_repeat - x_train)**2/(bandwidth ** 2) / 2, dim=1)
        y_hat = torch.matmul(attention_weights, y_train)
        return y_hat

    def pdf(self, bandwidth, x_train):
        n = x_train.size()[0]
        # data = x.unsqueeze(-2)
        # train_x = _unsqueeze_multiple_times(self.train_x, 0, len(s))

        data = self.x_test.repeat_interleave(n).reshape((-1, n))
        train_x = x_train.unsqueeze(0)
        # print(f'data={data.shape}')
        # print(f'train_x={train_x.shape}')

        pdf_values = (torch.exp(-((data - train_x) ** 2 / (bandwidth ** 2) / 2))
                     ).mean(dim=-1) / sqrt(2 * pi) / bandwidth

        return pdf_values


# def _unsqueeze_multiple_times(input, axis, times):
#     """
#     Utils function to unsqueeze tensor to avoid cumbersome code
#     :param input: A pytorch Tensor of dimensions (D_1,..., D_k)
#     :param axis: the axis to unsqueeze repeatedly
#     :param times: the number of repetitions of the unsqueeze
#     :return: the unsqueezed tensor. ex: dimensions (D_1,... D_i, 0,0,0, D_{i+1}, ... D_k) for unsqueezing 3x axis i.
#     """
#     output = input
#     for i in range(times):
#         output = output.unsqueeze(axis)
#     return output