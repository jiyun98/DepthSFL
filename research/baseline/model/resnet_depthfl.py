import math
import torch
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F
from math import ceil as up
from collections import OrderedDict

from model.blocks import *

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

''' ResNet18, 34 '''
# ---------------------------------------------------
#                split & pruned resnet
# ---------------------------------------------------
class ResNet18_client_v1(nn.Module):  # Dropout (or pruned) ResNet [width] 
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet18_client_v1, self).__init__()
        self.in_planes = 64
        
        # self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7,
        #                        stride=2, padding=3, bias=False)
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        out2 = self.layer1(out1)
        return [out2]

# ---------------------------------------------------
#                split & pruned resnet v2
# ---------------------------------------------------
class ResNet18_client_v2(nn.Module):  # Dropout (or pruned) ResNet [width] 
    def __init__(self, block, num_blocks,  num_classes=10):
        super(ResNet18_client_v2, self).__init__()
        self.in_planes = 64
        
        # self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7,
        #                        stride=2, padding=3, bias=False)
        # self.bn1 = nn.BatchNorm2d(self.in_planes)

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)

        self.apply(_weights_init)
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        return [out2, out3]
    

# ---------------------------------------------------
#                split & pruned resnet v5
# ---------------------------------------------------
class ResNet18_client_v3(nn.Module):  # Dropout (or pruned) ResNet [width] 
    def __init__(self, block, num_blocks,  num_classes=10):
        super(ResNet18_client_v3, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
         # self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)

        self.apply(_weights_init)
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        return [out2, out3, out4]


# for resnet18,34,50,101
class ResNet(nn.Module):  # Dropout (or pruned) ResNet [width] 
    def __init__(self, block, num_blocks,  num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
         # self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        self.apply(_weights_init)
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        out = F.avg_pool2d(out5, out5.size()[3])
        out = out.view(out.size(0), -1)
        logits = self.linear(out)
        probas = F.softmax(logits, dim=1)
        return [out2, out3, out4, logits], probas

''' ResNet50, 101'''
# ---------------------------------------------------
#                split & pruned resnet
# ---------------------------------------------------
class ResNet50_client_v1(nn.Module):  # Dropout (or pruned) ResNet [width] 
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet50_client_v1, self).__init__()
        self.in_planes = 64
        
        # self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7,
        #                        stride=2, padding=3, bias=False)
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        out2 = self.layer1(out1)
        return [out2]

# ---------------------------------------------------
#                split & pruned resnet v4
# ---------------------------------------------------
class ResNet50_client_v2(nn.Module):  # Dropout (or pruned) ResNet [width] 
    def __init__(self, block, num_blocks,  num_classes=10):
        super(ResNet50_client_v2, self).__init__()
        self.in_planes = 64
        
        # self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7,
        #                        stride=2, padding=3, bias=False)
        # self.bn1 = nn.BatchNorm2d(self.in_planes)

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)

        self.apply(_weights_init)
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        return [out2, out3]
    

# ---------------------------------------------------
#                split & pruned ResNet34 v5
# ---------------------------------------------------
class ResNet50_client_v3(nn.Module):  # Dropout (or pruned) ResNet34 [width] 
    def __init__(self, block, num_blocks,  num_classes=10):
        super(ResNet50_client_v3, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
         # self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)

        self.apply(_weights_init)
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        return [out2, out3, out4]
    
    
'''ResNet56, 110'''
# ---------------------------------------------------
#                split & pruned resnet
# ---------------------------------------------------
class ResNet56_client_v1(nn.Module): # Dropout (or pruned) ResNet [width] for CIFAR-10
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet56_client_v1, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        return [out]

#---------------------------------------------------------------------
class ResNet56_client_v2(nn.Module): # Dropout (or pruned) ResNet [width] for CIFAR-10
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet56_client_v2, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        return [out1, out2]
#----------------------------------------------------------------------
class ResNet56_client_v3(nn.Module): # Dropout (or pruned) ResNet [width] for CIFAR-10
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet56_client_v3, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        return [out1, out2, out3]


# for resnet56, 110
class ResNet_c(nn.Module): # Dropout (or pruned) ResNet [width] for CIFAR-10
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_c, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64*block.expansion, num_classes)
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out = F.avg_pool2d(out3, out3.size()[3])
        out = out.view(out.size(0), -1)
        logits = self.linear(out)
        probas = F.softmax(logits, dim=1)
        return [out1, out2, out3, logits], probas

'''resnet18'''
def resnet18_client(num_classes = 10, cut_point = 1):
    if cut_point == 1:
        model = ResNet18_client_v1(block = BasicBlock, num_blocks = [2,2,2,2],  num_classes = num_classes)
    elif cut_point == 2:
        model = ResNet18_client_v2(block = BasicBlock, num_blocks = [2,2,2,2],  num_classes = num_classes)
    elif cut_point == 3:
        model = ResNet18_client_v3(block = BasicBlock, num_blocks = [2,2,2,2], num_classes = num_classes)
    else:
        model = ResNet(block = BasicBlock, num_blocks = [2,2,2,2], num_classes = num_classes)
    return model

def resnet18_server(num_classes = 10, cut_point = 1):
    if cut_point == 1:
        model = ResNet18_server_v1(block = BasicBlock, num_blocks = [2,2,2,2], num_classes = num_classes)
    elif cut_point == 2:
        model = ResNet18_server_v2(block = BasicBlock, num_blocks = [2,2,2,2], num_classes = num_classes)
    elif cut_point == 3:
        model = ResNet18_server_v3(block = BasicBlock, num_blocks = [2,2,2,2], num_classes = num_classes)
    else:
        model = ResNet(block = BasicBlock, num_blocks = [2,2,2,2], num_classes = num_classes)
    return model

'''resnet34'''
def resnet34_client(num_classes = 10, cut_point = 1):
    if cut_point == 1:
        model = ResNet18_client_v1(block = BasicBlock, num_blocks = [3,4,6,3],  num_classes = num_classes)
    elif cut_point == 2:
        model = ResNet18_client_v2(block = BasicBlock, num_blocks = [3,4,6,3],  num_classes = num_classes)
    elif cut_point == 3:
        model = ResNet18_client_v3(block = BasicBlock, num_blocks = [3,4,6,3], num_classes = num_classes)
    else:
        model = ResNet(block = BasicBlock, num_blocks = [3,4,6,3], num_classes = num_classes)
    return model

def resnet34_server(num_classes = 10, cut_point = 1):
    if cut_point == 1:
        model = ResNet18_server_v1(block = BasicBlock, num_blocks = [3,4,6,3], num_classes = num_classes)
    elif cut_point == 2:
        model = ResNet18_server_v2(block = BasicBlock, num_blocks = [3,4,6,3], num_classes = num_classes)
    elif cut_point == 3:
        model = ResNet18_server_v3(block = BasicBlock, num_blocks = [3,4,6,3], num_classes = num_classes)
    return model


'''resnet50'''
def resnet50_client(num_classes = 10, cut_point = 1):
    if cut_point == 1:
        model = ResNet50_client_v1(block = BottleNeck, num_blocks = [3,4,6,3],  num_classes = num_classes)
    elif cut_point == 2:
        model = ResNet50_client_v2(block = BottleNeck, num_blocks = [3,4,6,3],  num_classes = num_classes)
    elif cut_point == 3:
        model = ResNet50_client_v3(block = BottleNeck, num_blocks = [3,4,6,3], num_classes = num_classes)
    else:
        model = ResNet(block = BottleNeck, num_blocks = [3,4,6,3],  num_classes = num_classes)
    return model

def resnet50_server(num_classes = 10, cut_point = 1):
    if cut_point == 1:
        model = ResNet50_server_v1(block = BottleNeck, num_blocks = [3,4,6,3], num_classes = num_classes)
    elif cut_point == 2:
        model = ResNet50_server_v2(block = BottleNeck, num_blocks = [3,4,6,3], num_classes = num_classes)
    elif cut_point == 3:
        model = ResNet50_server_v3(block = BottleNeck, num_blocks = [3,4,6,3], num_classes = num_classes)
    return model

'''resnet56'''
def resnet56_client(num_classes = 10, cut_point = 1):
    if cut_point == 1:
        model = ResNet56_client_v1(block = BasicBlock, num_blocks = [9,9,9],  num_classes = num_classes)
    elif cut_point == 2:
        model = ResNet56_client_v2(block = BasicBlock, num_blocks = [9,9,9],  num_classes = num_classes)
    elif cut_point == 3:
        model = ResNet56_client_v3(block = BasicBlock, num_blocks = [9,9,9], num_classes = num_classes)
    else:
        model = ResNet_c(block = BasicBlock, num_blocks = [9,9,9],  num_classes = num_classes)
    return model

def resnet56_server(num_classes = 10, cut_point = 1):
    if cut_point == 1:
        model = ResNet56_server_v1(block = BasicBlock, num_blocks = [9,9,9], num_classes = num_classes)
    elif cut_point == 2:
        model = ResNet56_server_v2(block = BasicBlock, num_blocks = [9,9,9], num_classes = num_classes)
    elif cut_point == 3:
        model = ResNet56_server_v3(block = BasicBlock, num_blocks = [9,9,9], num_classes = num_classes)
    return model


'''resnet101'''
def resnet101_client(num_classes = 10, cut_point = 1):
    if cut_point == 1:
        model = ResNet50_client_v1(block = BottleNeck, num_blocks = [3,4,23,3],  num_classes = num_classes)
    elif cut_point == 2:
        model = ResNet50_client_v2(block = BottleNeck, num_blocks = [3,4,23,3],  num_classes = num_classes)
    elif cut_point == 3:
        model = ResNet50_client_v3(block = BottleNeck, num_blocks = [3,4,23,3], num_classes = num_classes)
    else:
        model = ResNet(block = BottleNeck, num_blocks = [3,4,23,3],  num_classes = num_classes)
    return model

def resnet101_server(num_classes = 10, cut_point = 1):
    if cut_point == 1:
        model = ResNet50_server_v2(block = BottleNeck, num_blocks = [3,4,23,3], num_classes = num_classes)
    elif cut_point == 2:
        model = ResNet50_server_v2(block = BottleNeck, num_blocks = [3,4,23,3], num_classes = num_classes)
    elif cut_point == 3:
        model = ResNet50_server_v2(block = BottleNeck, num_blocks = [3,4,23,3], num_classes = num_classes)
    return model

'''resnet110'''
def resnet110_client(num_classes = 10, cut_point = 1):
    if cut_point == 1:
        model = ResNet56_client_v1(block = BasicBlock, num_blocks = [18,18,18],  num_classes = num_classes)
    elif cut_point == 2:
        model = ResNet56_client_v2(block = BasicBlock, num_blocks = [18,18,18],  num_classes = num_classes)
    elif cut_point == 3:
        model = ResNet56_client_v3(block = BasicBlock, num_blocks = [18,18,18], num_classes = num_classes)
    else:
        model = ResNet_c(block = BasicBlock, num_blocks = [18,18,18],  num_classes = num_classes)
    return model

def resnet110_server(num_classes = 10, cut_point = 1):
    if cut_point == 1:
        model = ResNet56_server_v1(block = BasicBlock, num_blocks = [18,18,18], num_classes = num_classes)
    elif cut_point == 2:
        model = ResNet56_server_v2(block = BasicBlock, num_blocks = [18,18,18], num_classes = num_classes)
    elif cut_point == 3:
        model = ResNet56_server_v3(block = BasicBlock, num_blocks = [18,18,18], num_classes = num_classes)
    return model