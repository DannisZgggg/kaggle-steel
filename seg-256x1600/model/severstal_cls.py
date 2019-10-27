import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision



class Net(nn.Module):
    def __init__(self,pretrained=False,num_class=4):
        super(Net, self).__init__()
        e = torchvision.models.resnet34(pretrained=pretrained)
        self.block = nn.ModuleList([
            e.conv1,
            e.bn1,
            e.relu,
            e.maxpool,
            e.layer1,
            e.layer2,
            e.layer3,
            e.layer4
        ])

        #e = None  #dropped
        self.feature = nn.Conv2d(512,32, kernel_size=1) #dummy conv for dim reduction
        self.logit = nn.Conv2d(32,num_class, kernel_size=1)

    def forward(self, x):
        batch_size,C,H,W = x.shape

        for i in range( len(self.block)):
            x = self.block[i](x)
            #print(i, x.shape)

        x = F.dropout(x,0.5,training=self.training)
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.feature(x)
        logit = self.logit(x)
        return logit.squeeze(-1).squeeze(-1)
