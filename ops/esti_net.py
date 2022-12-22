import torch.nn as nn
import torch
from ops.base_module import *

class ESTI_Net(nn.Module):

    def __init__(self, resnet_model, resnet_model1, apha, belta):
        super(ESTI_Net, self).__init__()

        self.conv1 = list(resnet_model.children())[0]
        self.bn1 = list(resnet_model.children())[1]
        self.relu = nn.ReLU(inplace=True)


        self.conv3d_1 = nn.Sequential(
            nn.Conv3d(in_channels=3, out_channels=3, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(3), nn.ReLU(inplace=True))
        self.conv3d_21 = nn.Sequential(
            nn.Conv3d(in_channels=3, out_channels=64, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(64), nn.ReLU(inplace=True))
        self.conv3d_22 = nn.Sequential(
            nn.Conv3d(in_channels=3, out_channels=64, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(0, 2, 2),
                      dilation=(1, 2, 2)),
            nn.BatchNorm3d(64), nn.ReLU(inplace=True))

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.layer1_bak = nn.Sequential(*list(resnet_model.children())[4])
        self.layer2_bak = nn.Sequential(*list(resnet_model.children())[5])
        self.layer3_bak = nn.Sequential(*list(resnet_model.children())[6])
        self.layer4_bak = nn.Sequential(*list(resnet_model.children())[7])
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.avg_diff = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc = list(resnet_model.children())[8]
        self.apha = apha
        self.belta = belta

    def forward(self, x):
        x1, x2, x3, x4, x5 = x[:, 0:3, :, :], x[:, 3:6, :, :], x[:, 6:9, :, :], x[:, 9:12, :, :], x[:, 12:15, :, :]

        nt, _, h, w = x.size()
        x_reshape = x.view(nt, 15, 1, h, w)
        x_reshape1, x_reshape2, x_reshape3, x_reshape4, x_reshape5 =x_reshape[:,0:3,:,:,:],x_reshape[:,3:6,:,:,:],x_reshape[:,6:9,:,:,:],x_reshape[:,9:12,:,:,:],x_reshape[:,12:15,:,:,:]
        x_reshape = torch.cat((x_reshape1, x_reshape2, x_reshape3, x_reshape4, x_reshape5), dim=2)  # n, c, t, h, w

        x_reshape = self.conv3d_1(x_reshape)
        x_reshape_1 = self.conv3d_21(x_reshape)
        x_reshape_2 = self.conv3d_22(x_reshape)
        x_reshape = 0.5*x_reshape_1 + 0.5*x_reshape_2

        x = self.conv1(x3)
        x = self.bn1(x)
        x = self.relu(x)
        # fusion layer1
        x = self.maxpool(x)

        x = self.apha * x + self.belta * x_reshape.squeeze(2)

        x = self.layer1_bak(x)
        x = self.layer2_bak(x)
        x = self.layer3_bak(x)
        x = self.layer4_bak(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x

def esti_net(base_model=None,num_segments=8,pretrained=True, **kwargs):
    if("18" in base_model):
        resnet_model = fbresnet18(num_segments, pretrained)
        resnet_model1 = fbresnet18(num_segments, pretrained)
    elif("50" in base_model):
        resnet_model = fbresnet50(num_segments, pretrained)
        resnet_model1 = fbresnet50(num_segments, pretrained)
    else:
        resnet_model = fbresnet101(num_segments, pretrained)
        resnet_model1 = fbresnet101(num_segments, pretrained)

    if(num_segments is 8):
        model = ESTI_Net(resnet_model,resnet_model1,apha=0.5,belta=0.5)
    else:
        model = ESTI_Net(resnet_model,resnet_model1,apha=0.75,belta=0.25)
    return model
