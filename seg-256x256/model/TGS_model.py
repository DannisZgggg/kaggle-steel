import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

using_syncbn = False
if not using_syncbn:
    BatchNorm2d = nn.BatchNorm2d
else:
    from model.sync_bn.nn import BatchNorm2dSync as BatchNorm2d


class FPAv2(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FPAv2, self).__init__()
        self.glob = nn.Sequential(nn.AdaptiveAvgPool2d((4,25)),  # nn.AdaptiveAvgPool2d(1)
                                  nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False))

        self.down2_1 = nn.Sequential(nn.Conv2d(input_dim, input_dim, kernel_size=5, stride=2, padding=2, bias=False),
                                     BatchNorm2d(input_dim),
                                     nn.ELU(True))
        self.down2_2 = nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=5, padding=2, bias=False),
                                     BatchNorm2d(output_dim),
                                     nn.ELU(True))

        self.down3_1 = nn.Sequential(nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=2, padding=1, bias=False),
                                     BatchNorm2d(input_dim),
                                     nn.ELU(True))
        self.down3_2 = nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1, bias=False),
                                     BatchNorm2d(output_dim),
                                     nn.ELU(True))

        self.conv1 = nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False),
                                   BatchNorm2d(output_dim),
                                   nn.ELU(True))

    def forward(self, x):
        # x shape: 512, 32, 200
        x_glob = self.glob(x)  # 256, 4 25
        x_glob = F.upsample(x_glob, scale_factor=8, mode='bilinear', align_corners=True)  # 256, 32, 200

        d2 = self.down2_1(x)  # 512, 8, 8
        d3 = self.down3_1(d2)  # 512, 4, 4

        d2 = self.down2_2(d2)  # 256, 8, 8
        d3 = self.down3_2(d3)  # 256, 4, 4

        d3 = F.upsample(d3, scale_factor=2, mode='bilinear', align_corners=True)  # 256, 8, 8
        d2 = d2 + d3

        d2 = F.upsample(d2, scale_factor=2, mode='bilinear', align_corners=True)  # 256, 16, 16
        x = self.conv1(x)  # 256, 16, 16
        x = x * d2
        x = x + x_glob  # 256, 32, 200
        return x


def conv3x3(input_dim, output_dim, rate=1):
    return nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=3, dilation=rate, padding=rate, bias=False),
                         BatchNorm2d(output_dim),
                         nn.ELU(True))


class SpatialAttention2d(nn.Module):
    def __init__(self, channel):
        super(SpatialAttention2d, self).__init__()
        self.squeeze = nn.Conv2d(channel, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = self.squeeze(x)
        z = self.sigmoid(z)
        return x * z


class GAB(nn.Module):
    def __init__(self, input_dim, reduction=4):
        super(GAB, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(input_dim, input_dim // reduction, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(input_dim // reduction, input_dim, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = self.global_avgpool(x)
        z = self.relu(self.conv1(z))
        z = self.sigmoid(self.conv2(z))
        return x * z


class Decoder(nn.Module):
    def __init__(self, in_channels, channels, out_channels):
        super(Decoder, self).__init__()
        self.conv1 = conv3x3(in_channels, channels)
        self.conv2 = conv3x3(channels, out_channels)
        self.s_att = SpatialAttention2d(out_channels)
        self.c_att = GAB(out_channels, 16)

    def forward(self, x, e=None):
        x = F.upsample(input=x, scale_factor=2, mode='bilinear', align_corners=True)
        if e is not None:
            x = torch.cat([x, e], 1)
        x = self.conv1(x)
        x = self.conv2(x)
        s = self.s_att(x)
        c = self.c_att(x)
        output = s + c
        return output


class Decoderv2(nn.Module):
    def __init__(self, up_in, x_in, n_out):
        super(Decoderv2, self).__init__()
        up_out = x_out = n_out // 2
        self.x_conv = nn.Conv2d(x_in, x_out, 1, bias=False)
        self.tr_conv = nn.ConvTranspose2d(up_in, up_out, 2, stride=2)
        self.bn = BatchNorm2d(n_out)
        self.relu = nn.ReLU(True)
        self.s_att = SpatialAttention2d(n_out)
        self.c_att = GAB(n_out, 16)

    def forward(self, up_p, x_p):
        up_p = self.tr_conv(up_p)
        x_p = self.x_conv(x_p)

        cat_p = torch.cat([up_p, x_p], 1)
        cat_p = self.relu(self.bn(cat_p))
        s = self.s_att(cat_p)
        c = self.c_att(cat_p)
        return s + c


class SCse(nn.Module):
    def __init__(self, dim):
        super(SCse, self).__init__()
        self.satt = SpatialAttention2d(dim)
        self.catt = GAB(dim)

    def forward(self, x):
        return self.satt(x) + self.catt(x)


# stage3 model
class SteelUnetv5(nn.Module):
    def __init__(self, n_classes, pretrained=True):
        super(SteelUnetv5, self).__init__()
        self.n_classes = n_classes
        #self.resnet = torchvision.models.resnet34(True)
        self.resnet = torchvision.models.resnet18(pretrained)

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            self.resnet.bn1,
            self.resnet.relu)

        self.encode2 = nn.Sequential(self.resnet.layer1,
                                     SCse(64))
        self.encode3 = nn.Sequential(self.resnet.layer2,
                                     SCse(128))
        self.encode4 = nn.Sequential(self.resnet.layer3,
                                     SCse(256))
        self.encode5 = nn.Sequential(self.resnet.layer4,
                                     SCse(512))

        self.center = nn.Sequential(FPAv2(512, 256),
                                    nn.MaxPool2d(2, 2))

        self.decode5 = Decoderv2(256, 512, 64)
        self.decode4 = Decoderv2(64, 256, 64)
        self.decode3 = Decoderv2(64, 128, 64)
        self.decode2 = Decoderv2(64, 64, 64)

        self.logit = nn.Sequential(nn.Conv2d(256, 32, kernel_size=3, padding=1),
                                   nn.ELU(True),
                                   nn.Conv2d(32, self.n_classes, kernel_size=1, bias=False))

    def forward(self, x):
        # x: batch_size, 3, 256, 1600
        x = self.conv1(x)  # 64, 256, 1600
        e2 = self.encode2(x)  # 64, 256, 1600
        e3 = self.encode3(e2)  # 128, 128, 800
        e4 = self.encode4(e3)  # 256, 64, 400
        e5 = self.encode5(e4)  # 512, 32, 200

        f = self.center(e5)  # 256, 16, 100

        d5 = self.decode5(f, e5)  # 64, 32, 200
        d4 = self.decode4(d5, e4)  # 64, 64, 400
        d3 = self.decode3(d4, e3)  # 64, 128, 800
        d2 = self.decode2(d3, e2)  # 64, 256, 1600

        f = torch.cat((d2,
                       F.upsample(d3, scale_factor=2, mode='bilinear', align_corners=True),
                       F.upsample(d4, scale_factor=4, mode='bilinear', align_corners=True),
                       F.upsample(d5, scale_factor=8, mode='bilinear', align_corners=True)), 1)  # 256, 128, 128

        f = F.dropout2d(f, p=0.4)  #0.4
        logit = self.logit(f)  # 1, 256, 1600

        return logit

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    img = torch.rand(4, 3, 256, 1600).cuda()
    net = SteelUnetv5(n_classes=4, pretrained=False)
    net = torch.nn.DataParallel(net)
    net = net.cuda()
    out = net(img)
    print(out.shape)
