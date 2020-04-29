from efficientdet.efficientdet import EfficientDet, Classifier, Instance
from dataset import EfficientdetDataset
from torch.utils.data import DataLoader
from config import get_args_efficientdet
from torchvision import transforms
from utils import Resizer, Normalizer, Augmenter, collater, AdamW
from efficientdet.loss import FocalLoss
import torch

opt = get_args_efficientdet()

# training_set = EfficientdetDataset(root_dir=opt.data_path, mode="train",
#                                transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]),
#                                imgORvdo=opt.imgORvdo, 
#                                maxLen=opt.max_position_embeddings,
#                                PAD=opt.PAD)

# loader = DataLoader(training_set, num_workers=2, batch_size=2, collate_fn=collater)
opt.num_classes = 90
net = EfficientDet(opt)
net.load_state_dict(torch.load('trained_models/efficientdet-d4.pth'))
net.classifier = Classifier(in_channels=224, num_anchors=9,
                                     num_classes=23,
                                     num_layers=4)
net.instance = Instance(in_channels=224, num_anchors=9, num_layers=4)
torch.save(net.state_dict(), 'trained_models/efficientdet-d4.pth')
# net.cuda()
# net.eval()
# cost = FocalLoss()

# for d in loader:
#     img = d['img'].cuda().float()
#     annot = d['annot'].cuda()
#     text = d['text'].cuda()
#     loss = net([img, text, annot])
#     print(loss)
#     break