

import sys
sys.path.append("/home/landelle/pytorch-cifar/models")
import resnet
import torch
import torch.nn as nn

# Wdie ResNet (7x7 conv window)
def resnet18(n=10):
    """ Get model is used to remake the model between lambda attempts """
    if n==10:
        return resnet.ResNet18()
    else:
        return resnet.ResNet(resnet.BasicBlock, [2,2,2,2], num_classes=n)

resnet18_100classes = lambda: resnet18(100)

# Fashion MNIST CNN
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
