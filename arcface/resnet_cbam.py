import torch
import torch.nn as nn
import math
from arcface.utils import l2_norm, Flatten, SentVec_TFIDF

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.ca = ChannelAttention(planes * 4)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

MODEL = {
    50:{
        'layers': [3, 4, 6, 3]
    },
    101:{
        'layers': [3, 4, 23, 3]
    },
    152:{
        'layers': [3, 8, 36, 3]
    }
}

class ResNetCBAM(nn.Module):

    def __init__(self, config):
        super(ResNetCBAM, self).__init__()
        embedding_size = config.embedding_size
        drop_ratio = config.drop_ratio
        model_dic = MODEL[config.num_layers_c]
        layers = model_dic['layers']
        # embedding_size = 2048
        # drop_ratio = 0.1
        # layers = [3, 4, 23, 3]

        # self.sentvec = SentVec_TFIDF(embedding_size=embedding_size, root_dir='data/')
        block = Bottleneck
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        # self.avgpool = nn.AvgPool2d(4, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512* block.expansion, 1000)

        self.bn_last = nn.BatchNorm1d(embedding_size)
        self.bn_last.bias.requires_grad_(False)

        # self.output_layer = nn.Sequential(
        #                             nn.BatchNorm2d(512 * block.expansion),
        #                             nn.Dropout(drop_ratio),
        #                             Flatten(),
        #                             nn.Linear(512 * block.expansion, embedding_size),
        #                             nn.BatchNorm1d(embedding_size))

        # self.last_layer = nn.Sequential(
        #     nn.Linear(2*embedding_size, embedding_size),
        #     nn.BatchNorm1d(embedding_size)
        # )
        '''if not config.resume:
            self._initialize_weights()                               
'''

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
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                X = stats.truncnorm(-2, 2, scale=0.01)
                values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                values = values.view(m.weight.size())
                with torch.no_grad():
                    m.weight.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        # print(x.size())
        x = self.layer2(x)
        # print(x.size())
        x = self.layer3(x)
        # print(x.size())
        x = self.layer4(x)
        # print(x.size())

        x = self.avgpool(x)
        # x = self.output_layer(x)
        # sent = self.sentvec(text)
        # x = torch.cat((x, sent), dim=1)
        # x = self.last_layer(x)
        x = torch.flatten(x, 1)
        
        if self.training:
            return x, self.bn_last(x)
        else:
            return l2_norm(self.bn_last(x))


if __name__ == "__main__":
    net = ResNetCBAM('aa')
    net.load_state_dict(torch.load('trained_models/resnet_cbam_101.pth'))
    # del net.output_layer
    # net.bn_last = nn.BatchNorm1d(2048)
    # l = [3, 4, 6, 3]
    # for i in range(3):
    #     net.layer1[i].ca = ChannelAttention(64 * 4)
    #     net.layer1[i].sa = SpatialAttention()
    # for i in range(4):
    #     net.layer2[i].ca = ChannelAttention(64 * 8)
    #     net.layer2[i].sa = SpatialAttention()
    # for i in range(6):
    #     net.layer3[i].ca = ChannelAttention(64 * 16)
    #     net.layer3[i].sa = SpatialAttention()
    # for i in range(3):
    #     net.layer4[i].ca = ChannelAttention(64 * 32)
    #     net.layer4[i].sa = SpatialAttention()
    
    # # net.sentvec = SentVec_TFIDF(embedding_size=512, root_dir='data/')
    # net.output_layer = nn.Sequential(
    #                             nn.BatchNorm2d(512* 4),
    #                             nn.Dropout(0.1),
    #                             Flatten(),
    #                             nn.Linear(512 * 4, 4096),
    #                             nn.BatchNorm1d(4096))
                                
    # del net.fc
    torch.save(net.state_dict(), 'trained_models/resnet_cbam_101.pth')
    a = torch.randn(5,3,224,224)
    b = net(a)
    print(b[0].size())