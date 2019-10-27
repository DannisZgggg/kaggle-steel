import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import numpy as np

def conv3x3(in_planes, out_planes, stride=1):
  "3x3 convolution with padding"
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
           padding=1, bias=False)


class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(BasicBlock, self).__init__()
    self.conv1 = conv3x3(inplanes, planes, stride)
    self.bn1 = nn.BatchNorm2d(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(planes, planes)
    self.bn2 = nn.BatchNorm2d(planes)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out

class ResNet(nn.Module):
  def __init__(self, layers):
    self.inplanes = 64
    super(ResNet, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                 bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True) # change
    self.layer1 = self._make_layer(BasicBlock, 64, layers[0])
    self.layer2 = self._make_layer(BasicBlock, 128, layers[1], stride=2)

    self.layer3 = self._make_layer(BasicBlock, 256, layers[2], stride=2)

    self.layer4 = self._make_layer(BasicBlock, 512, layers[3], stride=2)

    self.avgpool = nn.AvgPool2d(7)
    #self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


  def _make_layer(self, block, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * block.expansion,
              kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * block.expansion),
      )

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)

    return x

def resnet18(pretrained=True):
  """Constructs a ResNet-101 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  #model = ResNet(Bottleneck, [3, 4, 23, 3])
  model_path = 'weights/resnet18.pth'
  model = ResNet( [2, 2, 2, 2])
  if pretrained:
      state_dict = torch.load(model_path)
      model.load_state_dict({k: v for k, v in state_dict.items() if k in model.state_dict()})
      print('pre_trained model loaded')

  return model

class ConvBlock(nn.Module):
    """
    Helper module that consists of a Conv -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x

class Up(nn.Module):
    """
    Up block that encapsulates one up-sampling step which consists of Upsample -> ConvBlock -> ConvBlock
    """

    def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None,
                 upsampling_method="bilinear"):
        #TODO: upsampling_method
        super().__init__()

        if up_conv_in_channels == None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels == None:
            up_conv_out_channels = out_channels

        if upsampling_method == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
        elif upsampling_method == "bilinear":
            self.upsample = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                #nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
                # modified by zhang ge
                #CarafeUpsample(in_channels=up_conv_in_channels, scale_factor=2),
                #nn.Conv2d(up_conv_in_channels, up_conv_out_channels, kernel_size=3, stride=1,padding=1)
            )
        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

    def forward(self, x):
        """
        :param up_x: this is the output from the previous up block
        :param down_x: this is the output from the down block
        :return: upsampled feature map
        """
        #x_0, x_1 = x
        if x[1] is not None:
            x_0, x_1 = x
            x_0 = self.upsample(x_0)
            # print(np.shape(x))
            # print(np.shape(down_x))
            out = torch.cat([x_0, x_1], 1)
            out = self.conv_block_1(out)
            out = self.conv_block_2(out)
        else:
            x_0,_ = x
            x_0 = self.upsample(x_0)
            out = self.conv_block_1(x_0)
            out = self.conv_block_2(out)

        return out

class UNet(nn.Module):
    def __init__(self, n_classes,phase):
        super(UNet, self).__init__()
        self.resnet = resnet18(pretrained=False)
        self.RCNN_layer0 = nn.Sequential(self.resnet.conv1, self.resnet.bn1, self.resnet.relu)
        self.RCNN_layer1 = nn.Sequential(self.resnet.layer1)
        self.RCNN_layer2 = nn.Sequential(self.resnet.layer2)
        self.RCNN_layer3 = nn.Sequential(self.resnet.layer3)
        self.RCNN_layer4 = nn.Sequential(self.resnet.layer4)


        self.up1 = Up(in_channels=512 + 256,out_channels=256, up_conv_in_channels=512,
                      up_conv_out_channels=512)
        self.up2 = Up(in_channels=256 + 128, out_channels=128, up_conv_in_channels=256,
                      up_conv_out_channels=256)
        self.up3 = Up(in_channels=128+64,  out_channels=64, up_conv_in_channels=128,
                      up_conv_out_channels=128)
        self.up4 = Up(in_channels=64+64, out_channels=32, up_conv_in_channels=64,
                      up_conv_out_channels=64)
        self.up5 = Up(in_channels=32, out_channels=16, up_conv_in_channels=32,
                      up_conv_out_channels=32)
        self.out = nn.Conv2d(16, n_classes, kernel_size=1, stride=1)

        self._init_weights()

        if phase == 'train':
            model_path = 'weights/resnet18.pth'
            state_dict = torch.load(model_path)
            self.resnet.load_state_dict({k: v for k, v in state_dict.items() if k in self.resnet.state_dict()})
            print('pre_trained model loaded')

    def forward(self, x):
        #x1 = self.inc(x)
        x1 = self.RCNN_layer0(x)
        x2 = self.resnet.maxpool(x1)
        x2 = self.RCNN_layer1(x2)
        x3 = self.RCNN_layer2(x2)
        x4 = self.RCNN_layer3(x3)
        x5 = self.RCNN_layer4(x4)
        #x6 = self.pool(x5)
        #x6 = self.bridge(x6)
        #print('1', np.shape(x1))
        #print('2', np.shape(x2))
        #print('3', np.shape(x3))
        #print('4', np.shape(x4))
        #print('5', np.shape(x5))

        x = self.up1([x5, x4])
        #print('up1',np.shape(x))
        x = self.up2([x, x3])
        #print('up2', np.shape(x))
        x = self.up3([x, x2])
        #print('up3', np.shape(x))
        x = self.up4([x, x1])
        #print('up4', np.shape(x))
        x = self.up5([x,None])
        #print('up5', np.shape(x))
        #x = self.up6(x)
        #print('up6', np.shape(x))
        x = self.out(x)
        #print('out', np.shape(x))
        return x

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        def weights_init(m, mean, stddev, truncated=False):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)
            elif classname.find('BatchNorm') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

        normal_init(self.out,0,0.01)
        weights_init(self.up1,0,0.01)
        weights_init(self.up2, 0, 0.01)
        weights_init(self.up3, 0, 0.01)
        weights_init(self.up4, 0, 0.01)
        weights_init(self.up5, 0, 0.01)
        '''for p in self.RCNN_layer0[0].parameters(): p.requires_grad = False
        for p in self.RCNN_layer0[1].parameters(): p.requires_grad = False
        for p in self.RCNN_layer1.parameters(): p.requires_grad = False'''

        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters(): p.requires_grad = False

        #self.RCNN_layer0.apply(set_bn_fix)
        #self.RCNN_layer1.apply(set_bn_fix)
        #self.RCNN_layer2.apply(set_bn_fix)
        #self.RCNN_layer3.apply(set_bn_fix)
        #self.RCNN_layer4.apply(set_bn_fix)

if __name__ == '__main__':
    img = torch.rand(3,3,512,512)
    net = UNet(n_classes=4,phase='test')
    print(net)
