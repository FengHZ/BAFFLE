from collections import OrderedDict
import torch
from torch import nn


class SteinIdentity(nn.Module):
    def __init__(self, K=1000, sigma=1e-4, fd_format="forward"):
        super(SteinIdentity, self).__init__()
        self.K = K
        self.sigma = sigma
        self.fd_format = fd_format

    def _stein_estimation(self, input_image, label, true_loss):
        """
        :param input_image: the input data, usually with size B*C*H*W
        :param label: the target label of the input image
        :param true_loss: the loss with no perturbation parameters
        :param K: the sample number
        :param sigma: the sigma in perturbation
        :param filter_flag: whether we use multi-stage filter to accelerate gradient estimation
        :param split_ratio: the split ratio for filter
        :param quantile: the quantile for filter the valued parameters
        :return: parameter gradient with loss(input_image,label)
        """
        with torch.no_grad():
            original_parameter = self.state_dict()
            delta = OrderedDict()
            grad = OrderedDict()
            if self.fd_format == "forward":
                for name in original_parameter.keys():
                    delta[name] = torch.zeros(original_parameter[name].size()).cuda()
                    grad[name] = torch.zeros(original_parameter[name].size()).cuda()
                for i in range(self.K):
                    for name in delta.keys():
                        delta[name] = self.sigma * torch.randn(delta[name].size()).cuda()
                        original_parameter[name] += delta[name]
                    self.load_state_dict(original_parameter, strict=False)
                    _, loss_add_delta = self._forward(input_image, label)
                    difference = loss_add_delta - true_loss
                    for name in delta.keys():
                        grad[name] += (difference / self.sigma) * (delta[name] / self.sigma)
                        original_parameter[name] -= delta[name]
                # return parameter
                self.load_state_dict(original_parameter)
                for name, p in self.named_parameters():
                    p.grad = grad[name] / self.K
            elif self.fd_format == "center":
                for name in original_parameter.keys():
                    delta[name] = torch.zeros(original_parameter[name].size()).cuda()
                    grad[name] = torch.zeros(original_parameter[name].size()).cuda()
                for i in range(self.K):
                    for name in delta.keys():
                        delta[name] = self.sigma * torch.randn(delta[name].size()).cuda()
                        original_parameter[name] += delta[name]
                    self.load_state_dict(original_parameter, strict=False)
                    _, loss_add_delta = self._forward(input_image, label)
                    for name in delta.keys():
                        original_parameter[name] -= 2 * delta[name]
                    self.load_state_dict(original_parameter, strict=False)
                    _, loss_minus_delta = self._forward(input_image, label)
                    difference = loss_add_delta - loss_minus_delta
                    for name in delta.keys():
                        grad[name] += (difference / (2 * self.sigma)) * (delta[name] / self.sigma)
                        original_parameter[name] += delta[name]
                self.load_state_dict(original_parameter)
                for name, p in self.named_parameters():
                    p.grad = grad[name] / self.K
            else:
                raise NotImplementedError("Finite Difference Format {} is not implemented".format(self.fd_format))
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.3, norm_type="inf")
