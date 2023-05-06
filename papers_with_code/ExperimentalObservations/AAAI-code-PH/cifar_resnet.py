""" ResNet-18 architecture [1]_

Notes
-----
Relevant section : Experiments with PH

Relevant library : `PyTorch` [2]_

References
----------
.. [1] He, K.; Zhang, X.; Ren, S.; and Sun, J. 2016. Deep Residual Learning for Image Recognition.
   In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
.. [2] Paszke, A.; Gross, S.; Massa, F.; Lerer, A.; Bradbury, J.; Chanan, G.; Killeen, T.; 
   Lin, Z.; Gimelshein, N.; Antiga, L.; Desmaison, A.; Kopf, A.; Yang, E.; DeVito, Z.;
   Raison, M.; Tejani, A.; Chilamkurthy, S.; Steiner, B.; Fang, L.; Bai, J.; and Chintala, S.
   2019. PyTorch: An Imperative Style, High-PerformanceDeep Learning Library. In Wallach, H.;
   Larochelle, H.; Beygelzimer, A.; d'Alché-Buc, F.; Fox, E.; and Garnett, R., eds.,
   Advances in Neural Information Processing Systems 32, 8024–8035. Curran Associates, Inc.
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init

__all__ = ['ResNet', 'resnet18']

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A', ):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.r1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.r2 = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = self.r1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.r2(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10,):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        block_list = (
            self._make_layer(block, 64, num_blocks[0], stride=1, ),
            self._make_layer(block, 128, num_blocks[1], stride=2, ),
            self._make_layer(block, 256, num_blocks[2], stride=2, ),
            self._make_layer(block, 512, num_blocks[3], stride=2, ),
        )

        self.block_seq = nn.Sequential(*block_list)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        self.relu = nn.ReLU()

        self.activation = {}


    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output.cpu().detach()
        return hook

    def register_layer(self, layer, name):
        layer.register_forward_hook(self.get_activation(name))

    def _make_layer(self, block, planes, num_blocks, stride, sparsity=0.5, sparse_func='reg', bias=True):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride,))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.block_seq(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def test(net):
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))

