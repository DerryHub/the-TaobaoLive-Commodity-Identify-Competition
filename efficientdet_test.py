from efficientdet.efficientdet import *
from dataset import EfficientdetDataset
from torch.utils.data import DataLoader
from config import get_args_efficientdet
from torchvision import transforms
from utils import Resizer, Normalizer, Augmenter, collater, AdamW
from efficientdet.loss import FocalLoss
from efficientdet.flownet import FlowNetS
import torch
from torch import nn
opt = get_args_efficientdet()

# training_set = EfficientdetDataset(root_dir=opt.data_path, mode="train",
#                                transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]),
#                                imgORvdo=opt.imgORvdo, 
#                                maxLen=opt.max_position_embeddings,
#                                PAD=opt.PAD)

# loader = DataLoader(training_set, num_workers=2, batch_size=10, collate_fn=collater)
opt.num_classes = 23
net = EfficientDet(opt)
net.load_state_dict(torch.load('trained_models/efficientdet-d0_video_new.pth'))
# net.classifier = Classifier(in_channels=224, num_anchors=9,
#                                      num_classes=23,
#                                      num_layers=4)
# net.instance.header = SeparableConvBlock(64*3, 9, norm=False, activation=False)
# net.flownet = FlowNetS(batchNorm=False)
# net.bifpn_flow = nn.Sequential(
#             *[BiFPN(64,
#                     [512, 512, 1024],
#                     True if _ == 0 else False,
#                     attention=True)
#               for _ in range(3)])
# torch.save(net.state_dict(), 'trained_models/efficientdet-d0_video_new.pth')
# net.cuda()
# net.eval()
# cost = FocalLoss()

# for d in loader:
#     img = d['img'].cuda().float()
#     annot = d['annot'].cuda()
#     text = d['text'].cuda()
#     print(img.size())
#     b = net([img, text, annot])
#     print(len(b))
#     break