import math
import torch
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F
from math import ceil as up
from collections import OrderedDict

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
# ---------------------------------------------------
#                split & pruned resnet
# ---------------------------------------------------
class ResNet110_client_v1(nn.Module): # Dropout (or pruned) ResNet [width] for CIFAR-10
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet110_client_v1, self).__init__()
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
        out = self.layer1(out)
        out = self.layer2(out)
        return [out]

class ResNet110_server_v1(nn.Module): # Dropout (or pruned) ResNet [width] for CIFAR-10
    def __init__(self, block, num_blocks,  num_classes=10):
        super(ResNet110_server_v1, self).__init__()
        self.in_planes = 32

        layer2 : OrderdDict[str, nn.Module] = OrderedDict()
        self.layer2 = self._make_layer(block, 32, num_blocks[1], 1, layer2, block_index = 1, opt = True)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.fc = nn.Linear(64*block.expansion, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride, Olayer = None, block_index = None, opt = False):
        if opt:
            ind = block_index
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                Olayer[f"{int(ind)}"] = block(self.in_planes, planes, stride)
                self.in_planes = planes * block.expansion
                ind += 1
            return nn.Sequential(Olayer)
        else:
            strides = [stride] + [1]*(num_blocks-1)
            layers = []
            for stride in strides:
                layers.append(block(self.in_planes, planes, stride))
                self.in_planes = planes * block.expansion
            return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer2(x)
        out1 = self.layer3[0](out)
        out2 = self.layer3[1:8](out1)   
        out = self.layer3[8:](out2)                                    # <-----------------------------
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        logits = self.fc(out)
        probas = F.softmax(logits, dim=1)
        return [out1, out2, logits], probas
#---------------------------------------------------------------------
class ResNet110_client_v2(nn.Module): # Dropout (or pruned) ResNet [width] for CIFAR-10
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet110_client_v2, self).__init__()
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
        out = self.layer1(out)
        out1 = self.layer2[0](out)
        out = self.layer2[1:](out1)
        out = self.layer3(out)
        return [out1, out]

class ResNet110_server_v2(nn.Module): # Dropout (or pruned) ResNet [width] for CIFAR-10
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet110_server_v2, self).__init__()
        self.in_planes = 64

        layer3 : OrderdDict[str, nn.Module] = OrderedDict()
        self.layer3 = self._make_layer(block,64, num_blocks[2], 1, layer3, block_index = 1, opt = True)
        self.fc = nn.Linear(64*block.expansion, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride, Olayer = None, block_index = None, opt = False):
        if opt:
            ind = block_index
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                Olayer[f"{int(ind)}"] = block(self.in_planes, planes, stride)
                self.in_planes = planes * block.expansion
                ind += 1
            return nn.Sequential(Olayer)
        else:
            strides = [stride] + [1]*(num_blocks-1)
            layers = []
            for stride in strides:
                layers.append(block(self.in_planes, planes, stride))
                self.in_planes = planes * block.expansion
            return nn.Sequential(*layers)

    def forward(self, x):
        out1 = self.layer3[1:8](x)
        out = self.layer3[8:](out1)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        logits = self.fc(out)
        probas = F.softmax(logits, dim=1)
        return [out1, logits], probas

#---------------------------------------------------------------------
class ResNet110_client_v3(nn.Module): # Dropout (or pruned) ResNet [width] for CIFAR-10
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet110_client_v3, self).__init__()
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
        out = self.layer1(out)
        out1 = self.layer2[0](out)
        out2 = self.layer2[1:](out1)
        out3 = self.layer3[0](out2)
        out4 = self.layer3[1:](out3)
        return [out1, out3, out4]

class ResNet110_server_v3(nn.Module): # Dropout (or pruned) ResNet [width] for CIFAR-10
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet110_server_v3, self).__init__()
        self.in_planes = 64

        layer3 : OrderdDict[str, nn.Module] = OrderedDict()
        self.layer3 = self._make_layer(block,64, num_blocks[2], 1, layer3, block_index = 8, opt = True)
        self.fc = nn.Linear(64*block.expansion, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride, Olayer = None, block_index = None, opt = False):
        if opt:
            ind = block_index
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                Olayer[f"{int(ind)}"] = block(self.in_planes, planes, stride)
                self.in_planes = planes * block.expansion
                ind += 1
            return nn.Sequential(Olayer)
        else:
            strides = [stride] + [1]*(num_blocks-1)
            layers = []
            for stride in strides:
                layers.append(block(self.in_planes, planes, stride))
                self.in_planes = planes * block.expansion
            return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer3(x)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        logits = self.fc(out)
        probas = F.softmax(logits, dim=1)
        return [logits], probas