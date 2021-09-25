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
    def __init__(self, x_test_1, x_test_2):
        # self.train_x = x_train
        # self.train_y = y_train
        self.len_1 = len(x_test_1)
        self.len_2 = len(x_test_2)
        self.x_test_1 = x_test_1 ### two dimensional torch
        self.x_test_2 = x_test_2 ### two dimensional torch
        self.test_cart = torch.cartesian_prod(self.x_test_1, self.x_test_2)
    
    def forward(self, y_train, x_train, device_gpu):
        n = x_train.size()[0]
        # print(f'n={n}')
        d = 2
        bandwidth = torch.tensor((n * (d + 2) / 4.) ** (-1. / (d + 4))).to(device_gpu)

        y_hat = self.kde_regression(bandwidth, x_train, y_train)  ## m_1 * m_2
        y_mean = torch.mean(y_train)
        pdf_values = self.pdf(bandwidth, x_train) ## m_1 * m_2

        DP = torch.sum(torch.abs(y_hat-y_mean) * pdf_values) / torch.sum(pdf_values)
        return DP

    def kde_regression(self, bandwidth, x_train, y_train):
        n = x_train.size()[0]     
        X_repeat = self.test_cart.unsqueeze(2).repeat_interleave(n, dim=-1).permute(0, 2, 1) ## (m_1*m_2) * n * 2
        dist = torch.norm(X_repeat - x_train, dim=2) ## (m_1*m_2) * n
        attention_weights = nn.functional.softmax(-dist**2/(bandwidth ** 2) / 2, dim=1) ## (m_1*m_2) * n
        y_hat = torch.matmul(attention_weights, y_train) ##(m_1*m_2) * 1
        y_hat = y_hat.reshape((self.len_1, self.len_2))
        return y_hat

    def pdf(self, bandwidth, x_train):
        n = x_train.size()[0]
        # data = x.unsqueeze(-2)
        # train_x = _unsqueeze_multiple_times(self.train_x, 0, len(s))

        X_repeat = self.test_cart.unsqueeze(2).repeat_interleave(n, dim=-1) \
                    .permute(0, 2, 1) ## (m_1*m_2) * n * 2
        
        dist = torch.norm(X_repeat - x_train, dim=2) ## (m_1*m_2) * n

        pdf_values = (torch.exp(-( dist ** 2 / (bandwidth ** 2) / 2))
                     ).mean(dim=-1) / sqrt(2 * pi) / bandwidth
        pdf_values = pdf_values.reshape((self.len_1, self.len_2))
        return pdf_values
