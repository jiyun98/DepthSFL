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
class ResNet56_client_v1(nn.Module): # Dropout (or pruned) ResNet [width] for CIFAR-10
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet56_client_v1, self).__init__()
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
        return out

class ResNet56_server_v1(nn.Module): # Dropout (or pruned) ResNet [width] for CIFAR-10
    def __init__(self, block, num_blocks,  num_classes=10):
        super(ResNet56_server_v1, self).__init__()
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
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        logits = self.fc(out)
        probas = F.softmax(logits, dim=1)
        return logits, probas
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
        out = self.layer1(out)
        out = self.layer2(out)
        return out

class ResNet56_server_v2(nn.Module): # Dropout (or pruned) ResNet [width] for CIFAR-10
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet56_server_v2, self).__init__()
        self.in_planes = 32
        self.layer3 = self._make_layer(block,64, num_blocks[2], stride=2)
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
        return logits, probas
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
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        return out

class ResNet56_server_v3(nn.Module): # Dropout (or pruned) ResNet [width] for CIFAR-10
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet56_server_v3, self).__init__()
        self.in_planes = 64

        layer3 : OrderdDict[str, nn.Module] = OrderedDict()
        self.layer3 = self._make_layer(block,64, num_blocks[2], 1, layer3, block_index = 4, opt = True)
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
        return logits, probas

class ResNet56_fjord(nn.Module): # ResNetchp, Width-varying ResNet designed for CIFAR-10, less parameters with deeper network
    def __init__(self, block, num_blocks, p_drop, num_classes=10, track=False):
        super(ResNet56_fjord, self).__init__()
        self.in_planes = up(16*p_drop)

        self.conv1 = nn.Conv2d(3, up(16*p_drop), kernel_size=3,stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes, momentum=None, track_running_stats=track)
        self.layer1 = self._make_layer(block, up(16*p_drop), num_blocks[0], stride=1, track=track)
        self.layer2 = self._make_layer(block, up(32*p_drop), num_blocks[1], stride=2, track=track)
        self.layer3 = self._make_layer(block,up(64*p_drop), num_blocks[2], stride=2, track=track)
        self.fc = nn.Linear(up(64*block.expansion*p_drop), num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride, track):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, track))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        logits = self.fc(out)
        probas = F.softmax(logits, dim=1)
        return logits, probas