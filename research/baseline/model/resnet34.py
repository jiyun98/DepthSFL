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
class ResNet34_FL_v1(nn.Module):  # Dropout (or pruned) ResNet [width] 
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet34_FL_v1, self).__init__()
        self.in_planes = 64
        
        # self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7,
        #                        stride=2, padding=3, bias=False)
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
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
        out1 = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        out = self.layer1(out1)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        logits = self.linear(out)
        probas = F.softmax(logits, dim=1)
        return logits, probas

# ---------------------------------------------------
#                split & pruned resnet v2
# ---------------------------------------------------
class ResNet34_FL_v2(nn.Module):  # Dropout (or pruned) ResNet [width] 
    def __init__(self, block, num_blocks,  num_classes=10):
        super(ResNet34_FL_v2, self).__init__()
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
        self.linear = nn.Linear(128*block.expansion, num_classes)
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
        out = self.layer2(out2)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        logits = self.linear(out)
        probas = F.softmax(logits, dim=1)
        return logits, probas

# ---------------------------------------------------
#                split & pruned ResNet34 v3
# ---------------------------------------------------
class ResNet34_FL_v3(nn.Module):  # Dropout (or pruned) ResNet34 [width] 
    def __init__(self, block, num_blocks,  num_classes=10):
        super(ResNet34_FL_v3, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
         # self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.linear = nn.Linear(256*block.expansion, num_classes)
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
        out = F.avg_pool2d(out4, out4.size()[3])
        out = out.view(out.size(0), -1)
        logits = self.linear(out)
        probas = F.softmax(logits, dim=1)
        return logits, probas
# ---------------------------------------------------
#                Full FL
# ---------------------------------------------------
class ResNet34_FL(nn.Module):  # Dropout (or pruned) ResNet [width] 
    def __init__(self, block, num_blocks,  num_classes=10):
        super(ResNet34_FL, self).__init__()
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
        return logits, probas

# ---------------------------------------------------
#                FjORD
# ---------------------------------------------------
class ResNet34_fjord(nn.Module):  # Dropout (or pruned) ResNet [width] 
    def __init__(self, block, num_blocks,  p_drop, num_classes=10, track = False):
        super(ResNet34_fjord, self).__init__()
        self.in_planes = up(64*p_drop)        
        # self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7,
        #                        stride=2, padding=3, bias=False)
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes, momentum=None, track_running_stats=track)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, up(64*p_drop), num_blocks[0], stride=1, track=track)
        self.layer2 = self._make_layer(block, up(128*p_drop), num_blocks[1], stride=2, track=track)
        self.layer3 = self._make_layer(block, up(256*p_drop), num_blocks[2], stride=2, track=track)
        self.layer4 = self._make_layer(block, up(512*p_drop), num_blocks[3], stride=2, track=track)
        self.fc = nn.Linear(up(512*block.expansion*p_drop), num_classes)

        self.apply(_weights_init)
    def _make_layer(self, block, planes, num_blocks, stride, track):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        logits = self.fc(out)
        probas = F.softmax(logits, dim=1)
        return logits, probas

# ---------------------------------------------------
#                DepthFL
# ---------------------------------------------------

def up(value):
  return math.ceil(value)


# ---------------------------------------------------
#                 DepthFL
# ---------------------------------------------------
class ResNet34_depthFL_v1(nn.Module):  # Dropout (or pruned) ResNet [width] 
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet34_depthFL_v1, self).__init__()
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
#                V2
# ---------------------------------------------------
class ResNet34_depthFL_v2(nn.Module):  # Dropout (or pruned) ResNet [width] 
    def __init__(self, block, num_blocks,  num_classes=10):
        super(ResNet34_depthFL_v2, self).__init__()
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
#                V3
# ---------------------------------------------------
class ResNet34_depthFL_v3(nn.Module):  # Dropout (or pruned) ResNet [width] 
    def __init__(self, block, num_blocks,  num_classes=10):
        super(ResNet34_depthFL_v3, self).__init__()
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
    

class ResNet34_depthFL(nn.Module):  # Dropout (or pruned) ResNet [width] 
    def __init__(self, block, num_blocks,  num_classes=10):
        super(ResNet34_depthFL, self).__init__()
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