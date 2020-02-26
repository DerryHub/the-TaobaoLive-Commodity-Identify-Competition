import torch.nn as nn
import torch
import math
from src.efficientnet import EfficientNet
from src.utils import BBoxTransform, ClipBoxes, Anchors
from src.loss import FocalLoss
from src.config import EFFICIENTDET
# from config import EFFICIENTDET
# from efficientnet import EfficientNet
# from utils import BBoxTransform, ClipBoxes, Anchors
# from loss import FocalLoss
from torchvision.ops.boxes import nms as nms_torch


def nms(dets, thresh):
    return nms_torch(dets[:, :4], dets[:, 4], thresh)


class ConvBlock(nn.Module):
    def __init__(self, num_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, groups=num_channels),
            nn.Conv2d(num_channels, num_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=num_channels, momentum=0.9997, eps=4e-5), nn.ReLU())

    def forward(self, input):
        return self.conv(input)


class BiFPN(nn.Module):
    def __init__(self, num_channels, epsilon=1e-4):
        super(BiFPN, self).__init__()
        self.epsilon = epsilon
        # Conv layers
        self.conv6_up = ConvBlock(num_channels)
        self.conv5_up = ConvBlock(num_channels)
        self.conv4_up = ConvBlock(num_channels)
        self.conv3_up = ConvBlock(num_channels)
        self.conv4_down = ConvBlock(num_channels)
        self.conv5_down = ConvBlock(num_channels)
        self.conv6_down = ConvBlock(num_channels)
        self.conv7_down = ConvBlock(num_channels)

        # Feature scaling layers
        self.p6_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p5_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p4_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p3_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.p4_downsample = nn.MaxPool2d(kernel_size=2)
        self.p5_downsample = nn.MaxPool2d(kernel_size=2)
        self.p6_downsample = nn.MaxPool2d(kernel_size=2)
        self.p7_downsample = nn.MaxPool2d(kernel_size=2)

        # Weight
        self.p6_w1 = nn.Parameter(torch.ones(2))
        self.p6_w1_relu = nn.ReLU()
        self.p5_w1 = nn.Parameter(torch.ones(2))
        self.p5_w1_relu = nn.ReLU()
        self.p4_w1 = nn.Parameter(torch.ones(2))
        self.p4_w1_relu = nn.ReLU()
        self.p3_w1 = nn.Parameter(torch.ones(2))
        self.p3_w1_relu = nn.ReLU()

        self.p4_w2 = nn.Parameter(torch.ones(3))
        self.p4_w2_relu = nn.ReLU()
        self.p5_w2 = nn.Parameter(torch.ones(3))
        self.p5_w2_relu = nn.ReLU()
        self.p6_w2 = nn.Parameter(torch.ones(3))
        self.p6_w2_relu = nn.ReLU()
        self.p7_w2 = nn.Parameter(torch.ones(2))
        self.p7_w2_relu = nn.ReLU()

    def forward(self, inputs):
        """
            P7_0 -------------------------- P7_2 -------->
            P6_0 ---------- P6_1 ---------- P6_2 -------->
            P5_0 ---------- P5_1 ---------- P5_2 -------->
            P4_0 ---------- P4_1 ---------- P4_2 -------->
            P3_0 -------------------------- P3_2 -------->
        """
        # P3_0, P4_0, P5_0, P6_0 and P7_0
        p3_in, p4_in, p5_in, p6_in, p7_in = inputs
        # P7_0 to P7_2
        # Weights for P6_0 and P7_0 to P6_1
        p6_w1 = self.p6_w1_relu(self.p6_w1)
        weight = p6_w1 / (torch.sum(p6_w1, dim=0) + self.epsilon)
        # Connections for P6_0 and P7_0 to P6_1 respectively
        p6_up = self.conv6_up(weight[0] * p6_in + weight[1] * self.p6_upsample(p7_in))
        # Weights for P5_0 and P6_0 to P5_1
        p5_w1 = self.p5_w1_relu(self.p5_w1)
        weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)
        # Connections for P5_0 and P6_0 to P5_1 respectively
        p5_up = self.conv5_up(weight[0] * p5_in + weight[1] * self.p5_upsample(p6_up))
        # Weights for P4_0 and P5_0 to P4_1
        p4_w1 = self.p4_w1_relu(self.p4_w1)
        weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
        # Connections for P4_0 and P5_0 to P4_1 respectively
        p4_up = self.conv4_up(weight[0] * p4_in + weight[1] * self.p4_upsample(p5_up))

        # Weights for P3_0 and P4_1 to P3_2
        p3_w1 = self.p3_w1_relu(self.p3_w1)
        weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
        # Connections for P3_0 and P4_1 to P3_2 respectively
        p3_out = self.conv3_up(weight[0] * p3_in + weight[1] * self.p3_upsample(p4_up))

        # Weights for P4_0, P4_1 and P3_2 to P4_2
        p4_w2 = self.p4_w2_relu(self.p4_w2)
        weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
        # Connections for P4_0, P4_1 and P3_2 to P4_2 respectively
        p4_out = self.conv4_down(
            weight[0] * p4_in + weight[1] * p4_up + weight[2] * self.p4_downsample(p3_out))
        # Weights for P5_0, P5_1 and P4_2 to P5_2
        p5_w2 = self.p5_w2_relu(self.p5_w2)
        weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
        # Connections for P5_0, P5_1 and P4_2 to P5_2 respectively
        p5_out = self.conv5_down(
            weight[0] * p5_in + weight[1] * p5_up + weight[2] * self.p5_downsample(p4_out))
        # Weights for P6_0, P6_1 and P5_2 to P6_2
        p6_w2 = self.p6_w2_relu(self.p6_w2)
        weight = p6_w2 / (torch.sum(p6_w2, dim=0) + self.epsilon)
        # Connections for P6_0, P6_1 and P5_2 to P6_2 respectively
        p6_out = self.conv6_down(
            weight[0] * p6_in + weight[1] * p6_up + weight[2] * self.p6_downsample(p5_out))
        # Weights for P7_0 and P6_2 to P7_2
        p7_w2 = self.p7_w2_relu(self.p7_w2)
        weight = p7_w2 / (torch.sum(p7_w2, dim=0) + self.epsilon)
        # Connections for P7_0 and P6_2 to P7_2
        p7_out = self.conv7_down(weight[0] * p7_in + weight[1] * self.p7_downsample(p6_out))

        return p3_out, p4_out, p5_out, p6_out, p7_out


class Regressor(nn.Module):
    def __init__(self, in_channels, num_anchors, num_layers):
        super(Regressor, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(True))
        self.layers = nn.Sequential(*layers)
        self.header = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, stride=1, padding=1)

    def forward(self, inputs):
        inputs = self.layers(inputs)
        inputs = self.header(inputs)
        output = inputs.permute(0, 2, 3, 1)
        return output.contiguous().view(output.shape[0], -1, 4)


class Classifier(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes, num_layers):
        super(Classifier, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(True))
        self.layers = nn.Sequential(*layers)
        self.header = nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1)
        self.act = nn.Sigmoid()

    def forward(self, inputs):
        inputs = self.layers(inputs)
        inputs = self.header(inputs)
        inputs = self.act(inputs)
        inputs = inputs.permute(0, 2, 3, 1)
        # print(inputs.size())
        output = inputs.contiguous().view(inputs.shape[0], inputs.shape[1], inputs.shape[2], self.num_anchors,
                                          self.num_classes)
        return output.contiguous().view(output.shape[0], -1, self.num_classes)


class EfficientDet(nn.Module):
    def __init__(self, config):
        super(EfficientDet, self).__init__()
        self.is_training = config.is_training
        self.nms_threshold = config.nms_threshold
        model_conf = EFFICIENTDET[config.network]
        self.num_channels = model_conf['W_bifpn']
        input_channels = model_conf['EfficientNet_output']
        self.convs = []
        self.conv3 = nn.Conv2d(input_channels[0], self.num_channels, kernel_size=1, stride=1, padding=0)
        self.conv4 = nn.Conv2d(input_channels[1], self.num_channels, kernel_size=1, stride=1, padding=0)
        self.conv5 = nn.Conv2d(input_channels[2], self.num_channels, kernel_size=1, stride=1, padding=0)
        self.conv6 = nn.Conv2d(input_channels[3], self.num_channels, kernel_size=1, stride=1, padding=0)
        self.conv7 = nn.Conv2d(input_channels[4], self.num_channels, kernel_size=1, stride=1, padding=0)
        self.convs.append(self.conv3)
        self.convs.append(self.conv4)
        self.convs.append(self.conv5)
        self.convs.append(self.conv6)
        self.convs.append(self.conv7)

        self.bifpn = nn.Sequential(*[BiFPN(self.num_channels) for _ in range(model_conf['D_bifpn'])])

        self.num_classes = config.num_classes
        self.anchors = Anchors()
        self.regressor = Regressor(in_channels=self.num_channels, num_anchors=self.anchors.num_anchors,
                                   num_layers=model_conf['D_class'])
        self.classifier = Classifier(in_channels=self.num_channels, num_anchors=self.anchors.num_anchors, num_classes=self.num_classes,
                                     num_layers=model_conf['D_class'])

        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()
        self.focalLoss = FocalLoss()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        prior = 0.01

        self.classifier.header.weight.data.fill_(0)
        self.classifier.header.bias.data.fill_(-math.log((1.0 - prior) / prior))

        self.regressor.header.weight.data.fill_(0)
        self.regressor.header.bias.data.fill_(0)

        if config.resume:
            self.backbone_net = EfficientNet.from_name(model_conf['EfficientNet'])
        else:
            self.backbone_net = EfficientNet.from_pretrained(model_conf['EfficientNet'])

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def set_is_training(self, value):
        self.is_training = value

    def forward(self, inputs):
        if self.is_training:
            img_batch, annotations = inputs
        else:
            img_batch = inputs

        features = self.backbone_net(img_batch)[2:]
        # for p in features:
        #     print(p.size())
        for i, conv in enumerate(self.convs):
            features[i] = conv(features[i])

        features = self.bifpn(features)

        regression = torch.cat([self.regressor(feature) for feature in features], dim=1)
        classification = torch.cat([self.classifier(feature) for feature in features], dim=1)
        anchors = self.anchors(img_batch)
        # print(anchors.size())

        if self.is_training:
            # print(classification.size(), regression.size(), anchors.size(), annotations.size())
            return self.focalLoss(classification, regression, anchors, annotations)
        else:
            transformed_anchors = self.regressBoxes(anchors, regression)
            transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)
            # print(transformed_anchors.size())
            scores = torch.max(classification, dim=2, keepdim=True)[0]
            scores_over_thresh = (scores > 0.05)[:, :, 0]

            output_list = []
            batch_size = scores.size(0)
            for i in range(batch_size):

                # scores_over_thresh = (scores > 0.05)[i, :, 0]

                if scores_over_thresh[i, :].sum() == 0:
                    output_list.append([torch.zeros(0), torch.zeros(0), torch.zeros(0, 4)])
                    continue

                classification_i = classification[:, scores_over_thresh[i], :]
                transformed_anchors_i = transformed_anchors[:, scores_over_thresh[i], :]
                scores_i = scores[:, scores_over_thresh[i], :]

                anchors_nms_idx = nms(torch.cat([transformed_anchors_i, scores_i], dim=2)[i, :, :], self.nms_threshold)
                
                nms_scores, nms_class = classification_i[i, anchors_nms_idx, :].max(dim=1)
                output_list.append([nms_scores, nms_class, transformed_anchors_i[i, anchors_nms_idx, :]])
            return output_list


if __name__ == '__main__':
    import argparse
    def get_args():
        parser = argparse.ArgumentParser(
            "EfficientDet: Scalable and Efficient Object Detection implementation by Signatrix GmbH")
        parser.add_argument("--image_size", type=int, default=512, help="The common width and height for all images")
        parser.add_argument("--batch_size", type=int, default=14, help="The number of images per batch")
        parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument('--alpha', type=float, default=0.25)
        parser.add_argument('--gamma', type=float, default=1.5)
        parser.add_argument("--num_epochs", type=int, default=500)
        parser.add_argument("--test_interval", type=int, default=1, help="Number of epoches between testing phases")
        parser.add_argument("--es_min_delta", type=float, default=0.0,
                            help="Early stopping's parameter: minimum change loss to qualify as an improvement")
        parser.add_argument("--es_patience", type=int, default=0,
                            help="Early stopping's parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.")
        parser.add_argument("--data_path", type=str, default="data/coco", help="the root folder of dataset")
        parser.add_argument("--log_path", type=str, default="tensorboard/signatrix_efficientdet_coco")
        parser.add_argument("--saved_path", type=str, default="trained_models")

        parser.add_argument("--num_classes", type=int, default=80)
        parser.add_argument('--network', default='efficientdet-d0', type=str,
                            help='efficientdet-[d0, d1, ..]')
        parser.add_argument("--is_training", type=bool, default=True)                    
        args = parser.parse_args()
        return args
    config = get_args()
    # def count_parameters(model):
    #     return sum(p.numel() for p in model.parameters() if p.requires_grad)

    model = EfficientDet(config).cuda()
    # model = EffNet.from_pretrained('efficientnet-b0')
    # print(model)
    a = torch.randn([3,3,512,512]).cuda()
    model.set_is_training(False)
    model.eval()
    # b = torch.randn([3, 5, 5]).cuda()
    # c3, c4, c5 = model(a)
    print(model(a))
    # print(print(len(model._blocks)))
    # print(c3.size(), c4.size(), c5.size())
