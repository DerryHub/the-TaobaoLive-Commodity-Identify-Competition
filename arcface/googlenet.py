import torch
import torch.nn as nn
import torch.nn.functional as F
from arcface.utils import l2_norm, Flatten

class GoogLeNet(nn.Module):

    def __init__(self, config):
        super(GoogLeNet, self).__init__()
        drop_ratio = config.drop_ratio
        embedding_size = config.embedding_size

        # self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        # self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2 = BasicConv2d(3, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, stride=2)
        # self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64, stride=2)
        # self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64, stride=2)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128, stride=2)
        # self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128, stride=2)

        self.output_layer = nn.Sequential(nn.BatchNorm2d(1024), 
                                       nn.Dropout(drop_ratio),
                                       BasicConv2d(1024, 1024, kernel_size=4),
                                       Flatten(),
                                       nn.Linear(1024, embedding_size),
                                       nn.BatchNorm1d(embedding_size))

        if not config.resume:
            self._initialize_weights()

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
        # x = self.conv1(x)
        # x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # x = self.maxpool2(x)
        
        x = self.inception3a(x)
        x = self.inception3b(x)
        # x = self.maxpool3(x)
        x = self.inception4a(x)

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)

        x = self.inception4e(x)
        # x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.output_layer(x)
        return l2_norm(x)


class Inception(nn.Module):

    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj, stride=1):
        super(Inception, self).__init__()

        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1, stride=stride)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1, stride=stride)
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=3, padding=1, stride=stride)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            BasicConv2d(in_channels, pool_proj, kernel_size=1, stride=stride)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

if __name__ == "__main__":
    net = GoogLeNet()
    a = torch.randn(12, 3, 112, 112)
    b = net(a)
    print(b.size())