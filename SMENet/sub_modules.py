import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import cv2
import os
import numpy as np


# SAM
class SAM(nn.Module):
    def __init__(self, kernel_size=1, bias_sign=True):
        super(SAM, self).__init__()
        assert kernel_size % 2 == 1, "kernel_size = {}".format(kernel_size)
        padding = (kernel_size - 1) // 2
        self.layer = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=kernel_size, padding=padding, bias=bias_sign),
            nn.Sigmoid(),
        )

    def forward(self, x):
        avg_mask = torch.mean(x, dim=1, keepdim=True)   # W*H*1
        max_mask, _ = torch.max(x, dim=1, keepdim=True)     # W*H*1
        mask = torch.mul(avg_mask, max_mask)
        mask = self.layer(mask)
        return mask

class Get_mask(nn.Module):
    def __init__(self, input_channels=512):
        super(Get_mask, self).__init__()
        self.SAM = SAM(kernel_size=1)
        self.smax = nn.Sigmoid()

    def forward(self, x):
        spatial_fm = self.SAM(x)  # 1*W*H
        return spatial_fm

# upsample x：p, y:p+1
def Upsample(x, y):
    _, _, h1, w1 = x.size()
    result = F.upsample(y, size=(h1, w1), mode='bilinear')
    return result

# Erasure(input_channels=1024, out_padding=0 / input_channels=512, out_padding=2)
class Erasure(nn.Module):
    def __init__(self, input_channels=1024, channels=512):
        super(Erasure, self).__init__()
        self.get_mask = Get_mask()
        self.conv = nn.Conv2d(in_channels=input_channels, out_channels=512, kernel_size=1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, y):
        u_y = self.conv(y)
        u_y = Upsample(x, u_y)
        mask = self.get_mask(u_y)     # w*h*1
        e_x = x + torch.mul(x, (1.- mask))
        e_x = self.conv1(e_x)
        return e_x

def Erase():
    cfg = [(512, 512), (256, 1024)]
    erase = []
    for i in range(len(cfg)):
        era = Erasure(input_channels=cfg[i][0], channels=cfg[i][1])
        erase.append(era)
    return erase

# mask  (p2, p3')(p1, p3')(p1,p2')
class Get_fusion_mask_mul(nn.Module):
    def __init__(self, input_c=1024,input_channels=512):
        super(Get_fusion_mask_mul, self).__init__()
        self.get_mask = Get_mask(input_channels=input_c)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x, y):
        u_y = Upsample(x, y)
        c_x = self.conv1(x)
        mask = self.get_mask(u_y)
        result = torch.mul(c_x, mask)
        return result

# detailed information（p3,p13)(p3,p23)
class Fusion_detailed_information3(nn.Module):
    def __init__(self, in_places=512, places=256, stride=2):
        super(Fusion_detailed_information3, self).__init__()
        self.layer1 = nn.Sequential(OrderedDict(
            [
                ('conv1', nn.Conv2d(in_channels=in_places, out_channels=places, kernel_size=1, stride=1)),
                ('bn1', nn.BatchNorm2d(places)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv2', nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1)),
                ('bn2', nn.BatchNorm2d(places)),
                ('relu2', nn.ReLU(inplace=True)),
                ('conv3', nn.Conv2d(in_channels=places, out_channels=places*2, kernel_size=1, stride=1)),
                ('bn3', nn.BatchNorm2d(2*places)),
                ('relu3', nn.ReLU(inplace=True))
            ]
        ))
        self.layer2 = nn.Sequential(OrderedDict(
            [
                ('conv1', nn.Conv2d(in_channels=1536, out_channels=512, kernel_size=1, stride=1)),
                ('bn1', nn.BatchNorm2d(512)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv2', nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1)),
                ('bn2', nn.BatchNorm2d(256)),
                ('relu2', nn.ReLU(inplace=True)),
                ('conv3', nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=1)),
                ('bn3', nn.BatchNorm2d(512)),
                ('relu3', nn.ReLU(inplace=True))
            ]
        ))
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1, stride=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        self.get_fusion_mask_mul1 = Get_fusion_mask_mul(input_c=512, input_channels=1024)
        self.get_fusion_mask_mul2 = Get_fusion_mask_mul(input_c=512, input_channels=512)

    def forward(self, x1, x2, p):
        m1 = self.get_fusion_mask_mul1(x2, p)
        m2 = self.get_fusion_mask_mul2(x1, p)
        m2 = self.layer1(m2)
        m = torch.cat((m1, m2), dim=1)
        m = self.layer2(m)
        p = p + m
        result = self.layer3(p)
        return result

# detailed information（p2, p12)
class Fusion_detailed_information2(nn.Module):
    def __init__(self, in_places=512, places=512, stride=2):
        super(Fusion_detailed_information2, self).__init__()
        self.get_fusion_mask_mul = Get_fusion_mask_mul(input_channels=512)
        self.layer1 = nn.Sequential(OrderedDict(
            [
                ('conv1', nn.Conv2d(in_channels=in_places, out_channels=places, kernel_size=1, stride=1)),
                ('bn1', nn.BatchNorm2d(places)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv2', nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1)),
                ('bn2', nn.BatchNorm2d(places)),
                ('relu2', nn.ReLU(inplace=True)),
                ('conv3', nn.Conv2d(in_channels=places, out_channels=places * 2, kernel_size=1, stride=1)),
                ('bn3', nn.BatchNorm2d(places*2)),
                ('relu3', nn.ReLU(inplace=True))
            ]
        ))
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1, stride=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, p):
        m = self.get_fusion_mask_mul(x1, p)
        m = self.layer1(m)
        p = p + m
        result = self.layer2(p)
        return result

class Fusion_detailed_information1(nn.Module):
    def __init__(self):
        super(Fusion_detailed_information1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1, stride=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv1(x)
        return x

# offset
def Center_point_mapping(x, R):
    coordinate = []
    N, C, H, W = x.size()
    C = C - 1
    for k in range(N):
        point = []
        for i in range(1, H-1):
            for j in range(1, W-1):
                if (x[k, C, i, j] > x[k, C, i-1, j-1] and x[k, C, i, j] > x[k, C, i-1, j]
                        and x[k, C, i, j] > x[k, C, i-1, j+1] and x[k, C, i, j] > x[k, C, i, j-1]
                        and x[k, C, i, j] > x[k, C, i, j+1] and x[k, C, i, j] > x[k, C, i+1, j-1]
                        and x[k, C, i, j] > x[k, C, i+1, j] and x[k, C, i, j] > x[k, C, i+1, j+1]):
                    point.append([k, C, R*i, R*j])
                else:
                    continue
        coordinate.append(point)
    return coordinate

# coordinate transformation
def center_size(boxes):
    return torch.tensor((boxes[:, 2:] + boxes[:, :2])/2, requires_grad=False)  # cx, cy

# offset_loss
class Offset_loss(nn.Module):
    def __init__(self):
        super(Offset_loss, self).__init__()

    def forward(self, target, pre_offset):
        # center mapping
        c1 = Center_point_mapping(pre_offset[0], 4)
        c2 = Center_point_mapping(pre_offset[1], 8)
        c3 = Center_point_mapping(pre_offset[2], 16)

        point_sum = 0   # total number
        m = []   # center point set
        for k in range(len(c1)):
            num_1 = len(c1[k])
            num_2 = len(c2[k])
            num_3 = len(c3[k])
            point_sum += num_3 + num_2 + num_1
            v = c1[k] + c2[k] + c3[k]   # set of all center points of different sizes feature map
            p = torch.tensor(v, requires_grad=False)
            m.append(p)

        for j in range(len(m)):
            m[j] = torch.sum(m[j][:, 2:4], dim=0)
        center_sum = torch.stack(m)   # [4, 2]
        center_sum = center_sum.float()
        center_sum_t = torch.sum(center_sum, dim=0)  # [1, 2]

        N = len(target)
        n = []
        for i in range(N):
            truths = target[i][:, :-1].data
            truths = center_size(truths)   # [num_obj,2]
            n.append(truths)

        for j in range(len(n)):
            n[j] = torch.sum(n[j], dim=0)

        truths_sum = torch.stack(n)
        truths_sum_t = torch.sum(truths_sum, dim=0)  # [1, 2]

        off_loss = nn.SmoothL1Loss(size_average=False)

        off_x = off_loss(center_sum[:, 0], truths_sum[:, 0])
        off_y = off_loss(center_sum[:, 1], truths_sum[:, 1])
        loss = (((off_x / torch.abs(off_x)) * (center_sum_t[0] - truths_sum_t[0])) + (
                    (off_y / torch.abs(off_y)) * (center_sum_t[1] - truths_sum_t[1]))) / point_sum
        return loss

class Bottleneck(nn.Module):
    def __init__(self, in_places, places, expansion=4, downsampling=False, stride=1):
        super(Bottleneck, self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling
        self.conv1 = nn.Conv2d(in_channels=in_places, out_channels=places, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(places)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(places)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(in_channels=places, out_channels=places * self.expansion, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(places * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places*self.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(places*self.expansion)
            )
            for m in self.downsample.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight.data)
                if isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        out = self.relu3(x)
        if self.downsampling:
            residual = self.downsample(residual)
        out = out + residual
        return out

class ResNet_extraction(nn.Module):
    def __init__(self, blocks, expansion=4):
        super(ResNet_extraction, self).__init__()

        self.expansion = expansion
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, bias=False)
        #########################
        nn.init.xavier_uniform_(self.conv1.weight.data, gain=nn.init.calculate_gain('relu'))
        self.layer1 = self.make_layer(in_places=64, places=64, block=blocks[0], downsample=True, stride=2)
        self.layer2 = self.make_layer(in_places=256, places=128, block=blocks[1], downsample=True, stride=2)
        self.layer3 = self.make_layer(in_places=512, places=256, block=blocks[2], downsample=True, stride=2)
        self.layer4 = self.make_layer(in_places=512, places=256, block=blocks[3], downsample=True, stride=2)
        self.layer5 = self.make_layer(in_places=1024, places=256, block=blocks[4], downsample=True, stride=2)
        ##############################
        for m in self.layer4.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def make_layer(self, in_places, places, block, downsample, stride):
        layers = []
        layers.append(Bottleneck(in_places, places, downsampling=downsample, stride=stride))
        for i in range(1, block):
            layers.append(Bottleneck(places * self.expansion, places))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        b_ori = self.layer3(x)
        b = self.layer4(x)
        c_ori = self.layer5(b)
        return x, b_ori, c_ori


class ResNet_fusion(nn.Module):
    def __init__(self, blocks, expansion=4):
        super(ResNet_fusion, self).__init__()

        self.expansion = expansion
        self.layer6 = self.make_layer(in_places=1024, places=256, block=blocks[0], downsample=False,
                                      stride=1)
        self.layer7 = self.make_layer(in_places=1024, places=256, block=blocks[1], downsample=False,
                                      stride=1)
        self.layer8 = self.make_layer(in_places=1024, places=256, block=blocks[2], downsample=False,
                                      stride=1)

    def make_layer(self, in_places, places, block, downsample, stride):
        layers = []
        layers.append(Bottleneck(in_places, places, downsampling=downsample, stride=stride))
        for i in range(1, block):
            layers.append(Bottleneck(places * self.expansion, places))
        return nn.Sequential(*layers)

    def forward(self, x, y, z):
        x = self.layer6(x)
        y = self.layer7(y)
        z = self.layer8(z)
        return x, y, z

class Change_channels(nn.Module):
    def __init__(self):
        super(Change_channels, self).__init__()

        # self.layer9 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, bias=False)  # C9:W*H*512 => W*H*256
        # self.layer10 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, bias=False)  # C9:W*H*512 => W*H*256
        # self.layer11 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, bias=False)  # C11:W*H*512 => W*H*256

        # ##############################delete FBS#####################
        self.layer9 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1)  # C9:W*H*512 => W*H*256
        self.layer10 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1)  # C9:W*H*512 => W*H*256
        self.layer11 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1)  #

        # #############################only erase#####################
        # self.layer9 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)  # C9:W*H*512 => W*H*256
        # self.layer10 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1)  # C9:W*H*512 => W*H*256
        # self.layer11 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)  #
    def forward(self, x, y, z):
        x = self.layer9(x)
        y = self.layer10(y)
        z = self.layer11(z)
        return x, y, z

# value of back
def background(feature):
    N, C, H, W = feature.size()
    ker = torch.ones((N, C, H, W))
    for i in range(1, H-1):
        for j in range(1, W-1):
            ker[:, :, i, j] = 0
    ker = torch.cuda.FloatTensor(ker)
    m = torch.mul(feature, ker)
    m = torch.sum(m, dim=(2, 3))
    m = torch.div(m, 4*(H-1))
    m = torch.unsqueeze(torch.unsqueeze(m, dim=2), dim=3)
    m = m.expand_as(feature)
    return m

# membership fuction
def membership_function(delta):
    mask = 1. - torch.exp(-0.3*delta)
    x = torch.nonzero(mask < 0)   # Get the element index of x-bt < 0
    mask[x[:, 0], x[:, 1], x[:, 2], x[:, 3]] = 0  # Make sure that the element value in the mask is [0,1]
    return mask


# OSE
class OSE(nn.Module):
    def __init__(self, input_c=1024):
        super(OSE, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=input_c, out_channels=input_c, kernel_size=1, stride=1),
            nn.BatchNorm2d(input_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        r = self.layer(x)
        back_ground = background(r)   # BT
        delta = r - back_ground    # x-BT  (W*H*C)
        mask = membership_function(delta)
        result = r + torch.mul(r, mask)    # 512*W*H        return result


def OSE_():
    cfg1 = [1024, 1024, 1024]
    fbs = []
    for i in range(len(cfg1)):
        fb = OSE(input_c=cfg1[i])
        fbs.append(fb)
    return fbs
