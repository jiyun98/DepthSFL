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
class ResNet101_client_v1(nn.Module):  # Dropout (or pruned) ResNet [width] 
    def __init__(self, block, num_blocks,  num_classes=10):
        super(ResNet101_client_v1, self).__init__()
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
        return [out4]
    

class ResNet101_server_v1(nn.Module):  # Dropout (or pruned) ResNet101 [width] 
    def __init__(self, block, num_blocks,  num_classes=10):
        super(ResNet101_server_v1, self).__init__()
        self.in_planes = 1024
        
        # self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7,
        #                        stride=2, padding=3, bias=False)
        # self.bn1 = nn.BatchNorm2d(self.in_planes)
        layer3 : OrderdDict[str, nn.Module] = OrderedDict()
        self.layer3 = self._make_layer(block, 256, num_blocks[2], 1, layer3, block_index = 23 - num_blocks[2], opt = True)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
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
        out1 = self.layer3[1:9](x)
        out2 = self.layer3[9:20](out1)
        out = self.layer3[20:24](out2)
        out = self.layer4(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        logits = self.linear(out)
        probas = F.softmax(logits, dim=1)
        return [out1, out2, logits], probas

# ---------------------------------------------------
#                split & pruned resnet v4
# ---------------------------------------------------
class ResNet101_client_v2(nn.Module):  # Dropout (or pruned) ResNet [width] 
    def __init__(self, block, num_blocks,  num_classes=10):
        super(ResNet101_client_v2, self).__init__()
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
        return out4
    

class ResNet101_server_v2(nn.Module):  # Dropout (or pruned) ResNet101 [width] 
    def __init__(self, block, num_blocks,  num_classes=10):
        super(ResNet101_server_v2, self).__init__()
        self.in_planes = 1024
        
        # self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7,
        #                        stride=2, padding=3, bias=False)
        # self.bn1 = nn.BatchNorm2d(self.in_planes)
        layer3 : OrderdDict[str, nn.Module] = OrderedDict()
        self.layer3 = self._make_layer(block, 256, num_blocks[2], 1, layer3, block_index = 23 - num_blocks[2], opt = True)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
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
        out = self.layer4(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        logits = self.linear(out)
        probas = F.softmax(logits, dim=1)
        return logits, probas
