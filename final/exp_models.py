

import sys
sys.path.append("/home/landelle/pytorch-cifar/models")
import resnet
import torch
import torch.nn as nn

"""
Getter for the 10 classes version of ResNet18, for CIFAR10, if n unspecified.
If n is specified, returns the n classes version of ResNet18
"""
def resnet18(n=10):
    """ Get model is used to remake the model between lambda attempts """
    if n==10:
        return resnet.ResNet18()
    else:
        return resnet.ResNet(resnet.BasicBlock, [2,2,2,2], num_classes=n)

"""
Getter for the 100 classes version of ResNet18, for CIFAR100
"""
resnet18_100classes = lambda: resnet18(100)

""" Simple CNN with two convolution Layers + one FC layer, inspired by LeNet """
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7*7*32, 10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
