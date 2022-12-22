from __future__ import print_function, division, absolute_import
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

__all__ = ['FBResNet', 'fbresnet18', 'fbresnet50', 'fbresnet101']

model_urls = {
        'fbresnet18': 'http://data.lip6.fr/cadene/pretrainedmodels/resnet18-5c106cde.pth',
        'fbresnet50': 'http://data.lip6.fr/cadene/pretrainedmodels/resnet50-19c8e357.pth',
        'fbresnet101': 'http://data.lip6.fr/cadene/pretrainedmodels/resnet101-5d3b4d8f.pth'
}


class MSTEModule(nn.Module):        #multi_spatial_temperal_module
    def __init__(self, channel, path, num_segments=8):
        super(MSTEModule, self).__init__()
        self.num_segments = num_segments
        self.channel = channel
        self.reduction = channel
        self.path = path

        self.p1_conv = nn.Conv3d(in_channels=1,
                                 out_channels=1,
                                 kernel_size=(3, 3, 3),
                                 stride=(1, 1 ,1),
                                 bias=False,
                                 padding=(1, 1, 1))
        self.p1_bn = nn.BatchNorm3d(1)

        if self.path > 1:
            self.p2_avg_pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.p2_conv = nn.Conv3d(in_channels=1,
                                     out_channels=1,
                                     kernel_size=(3, 3, 3),
                                     stride=(1, 1, 1),
                                     bias=False,
                                     padding=(1, 1, 1))
            self.p2_bn = nn.BatchNorm3d(1)

        if self.path > 2:
            self.p3_avg_pool4 = nn.AvgPool2d(kernel_size=4, stride=4)
            self.p3_conv = nn.Conv3d(in_channels=1,
                                      out_channels=1,
                                      kernel_size=(3, 3, 3),
                                      stride=(1, 1, 1),
                                      bias=False,
                                      padding=(1, 1, 1))
            self.p3_bn = nn.BatchNorm3d(1)

        if self.path > 3:
            self.p4_avg_pool8 = nn.AvgPool2d(kernel_size=8, stride=8)
            self.p4_conv = nn.Conv3d(in_channels=1,
                                      out_channels=1,
                                      kernel_size=(3, 3, 3),
                                      stride=(1, 1, 1),
                                      bias=False,
                                      padding=(1, 1, 1))
            self.p4_bn = nn.BatchNorm3d(1)

        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        nt, c, h, w = x.size()
        n = nt // self.num_segments

        x_reduction = x.mean(1, keepdim=True)     #(nt, 1, h, w)

        x_p1 = x_reduction        #nt, 1, h, w
        if self.path > 1:
            x_p2 = self.p2_avg_pool2(x_reduction)
        if self.path > 2:
            x_p3 = self.p3_avg_pool4(x_reduction)
        if self.path > 3:
            x_p4 = self.p4_avg_pool8(x_reduction)

        #reshape:n, 1, t, h, w
        x_p1 = x_p1.view(n, self.num_segments, 1, h, w).transpose(2, 1).contiguous()
        if self.path > 1:
            x_p2 = x_p2.view(n, self.num_segments, 1, h//2, w//2).transpose(2, 1).contiguous()
        if self.path > 2:
            x_p3 = x_p3.view(n, self.num_segments, 1, h//4, w//4).transpose(2, 1).contiguous()
        if self.path > 3:
            x_p4 = x_p4.view(n, self.num_segments, 1, h//8, w//8).transpose(2, 1).contiguous()

        x_p1 = self.p1_bn(self.p1_conv(x_p1))
        if self.path > 1:
            x_p2 = self.p2_bn(self.p2_conv(x_p2))
            x_p2 = F.interpolate(x_p2, x_p1.size()[2:])
        if self.path > 2:
            x_p3 = self.p3_bn(self.p3_conv(x_p3))
            x_p3 = F.interpolate(x_p3, x_p1.size()[2:])
        if self.path > 3:
            x_p4 = self.p4_bn(self.p4_conv(x_p4))
            x_p4 = F.interpolate(x_p4, x_p1.size()[2:])

        if self.path == 4:
            y = 1.0/4.0*x_p1 + 1.0/4.0*x_p2 + 1.0/4.0*x_p3 + 1.0/4.0*x_p4
        elif self.path == 3:
            y = 1.0/3.0*x_p1 + 1.0/3.0*x_p2 + 1.0/3.0*x_p3
        elif self.path == 2:
            y = 1.0/2.0*x_p1 + 1.0/2.0*x_p2
        else:
            y = x_p1

        y = y.transpose(2,1).contiguous().view(nt, -1, h, w)

        y = self.sigmoid(y) - 0.5
        output = x * y
        return output


def conv_3x1x1_bn(inp, oup, groups=1, identity=False):
    if identity:
        return nn.Sequential(
            nn.Conv3d(inp, oup, (3, 1, 1), (1, 1, 1), (1, 0, 0), groups=groups, bias=False),
            nn.BatchNorm3d(oup),
        )
    else:
        return nn.Sequential(
            nn.Conv3d(inp, oup, (3, 1, 1), (1, 1, 1), (1, 0, 0), groups=groups, bias=False),
            nn.BatchNorm3d(oup),
            nn.ReLU(inplace=True)
        )

def conv_3x1x1(inp, oup, groups=1):
    return nn.Sequential(
        nn.Conv3d(inp, oup, (3, 1, 1), (1, 1, 1), (1, 0, 0), groups=groups, bias=False),
    )

class MultiScaleTemporalAttention(nn.Module):
    def __init__(self, planes, num_segments=8, div=4):
        super(MultiScaleTemporalAttention, self).__init__()
        self.c = planes // div
        self.num_segments = num_segments
        self.fc1 = conv_3x1x1_bn(planes, self.c)
        self.fc2 = conv_3x1x1_bn(self.c, self.c)
        self.fc3 = conv_3x1x1(self.c, planes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        nt, c, h, w = x.size()
        n = nt // self.num_segments
        x = x.view(n, self.num_segments, c, h, w).contiguous().permute(0, 2, 1, 3, 4) #n, c, t, h, w

        out = F.avg_pool3d(x, kernel_size=[1, h, w])  #n, c, t, 1, 1

        out = self.fc1(out)
        outscale1 = out
        out = self.fc2(out)
        out = out + outscale1
        out = self.fc3(out)
        out = self.sigmoid(out) - 0.5

        out = x * out   #n, c, t, h, w

        out = out.permute(0, 2, 1, 3, 4).contiguous().view(nt, c, h, w)
        return out

class ShiftModule(nn.Module):
    def __init__(self, input_channels, n_segment=8,n_div=8, mode='shift'):
        super(ShiftModule, self).__init__()
        self.input_channels = input_channels
        self.n_segment = n_segment
        self.fold_div = n_div
        self.fold = self.input_channels // self.fold_div
        self.conv = nn.Conv1d(self.fold_div*self.fold, self.fold_div*self.fold,
                kernel_size=3, padding=1, groups=self.fold_div*self.fold,
                bias=False)

        if mode == 'shift':
            self.conv.weight.requires_grad = True
            self.conv.weight.data.zero_()
            self.conv.weight.data[:self.fold, 0, 2] = 1 # shift left
            self.conv.weight.data[self.fold: 2 * self.fold, 0, 0] = 1 # shift right
            if 2*self.fold < self.input_channels:
                self.conv.weight.data[2 * self.fold:, 0, 1] = 1 # fixed
        elif mode == 'fixed':
            self.conv.weight.requires_grad = True
            self.conv.weight.data.zero_()
            self.conv.weight.data[:, 0, 1] = 1 # fixed
        elif mode == 'norm':
            self.conv.weight.requires_grad = True

    def forward(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w)
        x = x.permute(0, 3, 4, 2, 1) # (n_batch, h, w, c, n_segment)
        x = x.contiguous().view(n_batch*h*w, c, self.n_segment)
        x = self.conv(x) # (n_batch*h*w, c, n_segment)
        x = x.view(n_batch, h, w, c, self.n_segment)
        x = x.permute(0, 4, 3, 1, 2) # (n_batch, n_segment, c, h, w)
        x = x.contiguous().view(nt, c, h, w)
        return x

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, num_segments, inplanes, planes, path_num, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        self.num_segments = num_segments

        self.t_attention = MultiScaleTemporalAttention(planes=planes, div=4)
        self.mste = MSTEModule(planes, path_num)

        self.shift = ShiftModule(planes, n_segment=self.num_segments, n_div=8, mode='shift')

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out1 = self.mste(out)
        out2 = self.t_attention(out)

        out = out + out1 + out2
        out = self.shift(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, num_segments, inplanes, planes, path_num, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.num_segments = num_segments

        self.t_attention = MultiScaleTemporalAttention(planes=planes, div=4)
        self.mste = MSTEModule(planes, path_num)

        self.shift = ShiftModule(planes, n_segment=self.num_segments, n_div=8, mode='shift')

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=True)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out1 = self.mste(out)
        out2 = self.t_attention(out)

        out = out + out1 + out2
        out = self.shift(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out

class FBResNet(nn.Module):

    def __init__(self, num_segments, block, layers, num_classes=1000):
        super(FBResNet, self).__init__()
        self.inplanes = 64

        self.input_space = None
        self.input_size = (224, 224, 3)
        self.mean = None
        self.std = None
        self.num_segments = num_segments

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                bias=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(self.num_segments,block, 64, layers[0], 4)
        self.layer2 = self._make_layer(self.num_segments,block, 128, layers[1], 3, stride=2)
        self.layer3 = self._make_layer(self.num_segments,block, 256, layers[2], 2, stride=2)
        self.layer4 = self._make_layer(self.num_segments,block, 512, layers[3], 1, stride=2)
        self.last_linear = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _make_layer(self, num_segments ,block, planes, blocks, path_num, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(num_segments, self.inplanes, planes, path_num, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(num_segments, self.inplanes, planes, path_num))

        return nn.Sequential(*layers)


    def features(self, input):
        x = self.conv1(input)
        self.conv1_input = x.clone()
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def logits(self, features):
        adaptiveAvgPoolWidth = features.shape[2]
        x = F.avg_pool2d(features, kernel_size=adaptiveAvgPoolWidth)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x


    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


def fbresnet18(num_segments=8,pretrained=False,num_classes=1000):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = FBResNet(num_segments,BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    if pretrained:
         model.load_state_dict(model_zoo.load_url(model_urls['fbresnet18']),strict=False)
    return model

def fbresnet50(num_segments=8,pretrained=False,num_classes=1000):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = FBResNet(num_segments,Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
    if pretrained:
         model.load_state_dict(model_zoo.load_url(model_urls['fbresnet50']),strict=False)
    return model


def fbresnet101(num_segments,pretrained=False,num_classes=1000):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = FBResNet(num_segments,Bottleneck, [3, 4, 23, 3], num_classes=num_classes)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['fbresnet101']),strict=False)
    return model
