import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
#from model_resnext50 import resnext50

using_syncbn = True
if not using_syncbn:
    BatchNorm2d = nn.BatchNorm2d
else:
    #from .sync_bn.nn import BatchNorm2dSync as BatchNorm2d
    from apex.parallel import SyncBatchNorm as BatchNorm2d


class ConvBn2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1, stride=1, groups=1, is_bn=True):
        super(ConvBn2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                              dilation=dilation, groups=groups, bias=False)
        # self.bn = SynchronizedBatchNorm2d(out_channels)
        self.bn = BatchNorm2d(out_channels)
        self.activate = nn.ELU(True)

    def forward(self, z):
        x = self.conv(z)
        x = self.bn(x)
        x = self.activate(x)
        return x


class sSE(nn.Module):
    def __init__(self, channel):
        super(sSE, self).__init__()
        self.squeeze = nn.Conv2d(channel, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = self.squeeze(x)
        z = self.sigmoid(z)
        return x * z

class cSE(nn.Module):
    def __init__(self, input_dim, reduction=4):
        super(cSE, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d((2,12))
        self.conv1 = nn.Conv2d(input_dim, input_dim // reduction, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(input_dim // reduction, input_dim, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = self.global_avgpool(x)
        z = self.relu(self.conv1(z))
        z = self.sigmoid(self.conv2(z))
        #added<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<,
        z = F.upsample(z, size=x.shape[2:], mode='bilinear', align_corners=True)
        return x * z

class SCse(nn.Module):
    def __init__(self, dim):
        super(SCse, self).__init__()
        self.satt = sSE(dim)
        self.catt = cSE(dim)

    def forward(self, x):
        return self.satt(x) + self.catt(x)

class Decoder(nn.Module):
    def __init__(self, in_channels, channels, out_channels):
        super(Decoder, self).__init__()
        self.conv1 = ConvBn2d(in_channels, channels, kernel_size=3, padding=1)
        self.conv2 = ConvBn2d(channels, out_channels, kernel_size=3, padding=1)
        self.att = SCse(out_channels)

    def forward(self, x, e=None):
        x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        # print('x',x.size())
        # print('e',e.size())
        if e is not None:
            x = torch.cat([x, e], 1)

        x = nn.ELU(True)(self.conv1(x))
        x = nn.ELU(True)(self.conv2(x))
        x = self.att(x)
        return x


class Unet_scSE_hyper(nn.Module):

    def __init__(self,n_classes=4, pretrained=False):
        super().__init__()
        self.n_classes = n_classes

        #torchvision models:
        self.resnet = torchvision.models.resnet18(pretrained=pretrained)
        self.conv1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
        )
        self.encoder2 = nn.Sequential(self.resnet.layer1,
                                     SCse(64))
        self.encoder3 = nn.Sequential(self.resnet.layer2,
                                     SCse(128))
        self.encoder4 = nn.Sequential(self.resnet.layer3,
                                     SCse(256))
        self.encoder5 = nn.Sequential(self.resnet.layer4,
                                     SCse(512))

        '''#resnext models:
        self.resnet = resnext50(pretrained=pretrained)
        print(self.resnet)
        self.conv1 = nn.Sequential(
            self.resnet.layer0.conv1,
            self.resnet.layer0.bn1,
            self.resnet.layer0.relu1
        )

        self.encoder2 = self.resnet.layer1
        self.encoder3 = self.resnet.layer2
        self.encoder4 = self.resnet.layer3
        self.encoder5 = self.resnet.layer4'''


        self.center = nn.Sequential(
            ConvBn2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.decoder5 = Decoder(256 + 512, 512, 64)
        self.decoder4 = Decoder(64 + 256, 256, 64)
        self.decoder3 = Decoder(64 + 128, 128, 64)
        self.decoder2 = Decoder(64 + 64, 64, 64)
        self.decoder1 = Decoder(64, 32, 64)

        self.logit = nn.Sequential(
            nn.Conv2d(384, 64, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(64, self.n_classes, kernel_size=1, padding=0),
        )

    def forward(self, x):

        e1 = self.conv1(x) #64, 128, 800
        #print('e1',e1.size())
        e2 = self.encoder2(e1) #64, 128, 800
        #print('e2',e2.size())
        e3 = self.encoder3(e2) #128, 64, 400
        #print('e3',e3.size())
        e4 = self.encoder4(e3) #256, 32, 200
        #print('e4',e4.size())
        e5 = self.encoder5(e4) #512, 16, 100
        #print('e5',e5.size())

        f = self.center(e5) #256, 8, 50
        #print('f',f.size())
        d5 = self.decoder5(f, e5)
        d4 = self.decoder4(d5, e4)
        d3 = self.decoder3(d4, e3)
        d2 = self.decoder2(d3, e2)
        d1 = self.decoder1(d2)
        #print('d1',d1.size())

        f = torch.cat((
            F.upsample(e1, scale_factor=2, mode='bilinear', align_corners=False),
            d1,
            F.upsample(d2, scale_factor=2, mode='bilinear', align_corners=False),
            F.upsample(d3, scale_factor=4, mode='bilinear', align_corners=False),
            F.upsample(d4, scale_factor=8, mode='bilinear', align_corners=False),
            F.upsample(d5, scale_factor=16, mode='bilinear', align_corners=False),
        ), 1)

        f = F.dropout2d(f, p=0.50)
        logit = self.logit(f)
        return logit


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    img = torch.rand(2, 3, 256, 1600)
    net = Unet_scSE_hyper(n_classes=4, pretrained=True)
    #net = torch.nn.DataParallel(net)
    net = net
    out = net(img)
    print('out ',out.shape)
