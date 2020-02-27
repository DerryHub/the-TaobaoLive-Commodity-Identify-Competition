import os
import argparse
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import EfficientdetDataset
from utils import Resizer, Normalizer, collater
from efficientdet.config import colors
import cv2
import shutil
from efficientdet.efficientdet import EfficientDet
from config import get_args_efficientdet
from tqdm import tqdm

writePIC = True

def iou(a, b):
    a = torch.clamp(a.long(), 0, 511)
    b = torch.clamp(b.long(), 0, 511)
    img_a = torch.zeros([512, 512])
    img_b = torch.zeros([512, 512])
    for t in a:
        img_a[t[0]:t[2], t[1]:t[3]] = 1
    for t in b:
        img_b[t[0]:t[2], t[1]:t[3]] = 1
    intersection = img_a*img_b
    ua = torch.clamp(img_a+img_b, max=1)
    return (intersection.sum()+1e-8) / (ua.sum()+1e-8)
    

def test(opt):
    opt.resume = True
    test_set = EfficientdetDataset(opt.data_path, mode='validation', transform=transforms.Compose([Normalizer(), Resizer()]))
    opt.num_classes = test_set.num_classes
    opt.batch_size = opt.batch_size*4
    test_params = {"batch_size": opt.batch_size,
                   "shuffle": False,
                   "drop_last": False,
                   "collate_fn": collater,
                   "num_workers": 12}
    test_generator = DataLoader(test_set, **test_params)
    
    model = EfficientDet(opt)
    model.load_state_dict(torch.load(os.path.join(opt.saved_path, opt.network+'.pth')))
    model.cuda()
    model.set_is_training(False)
    model.eval()

    if os.path.isdir(opt.prediction_dir):
        shutil.rmtree(opt.prediction_dir)
    os.makedirs(opt.prediction_dir)
    
    progress_bar = tqdm(test_generator)
    progress_bar.set_description_str(' Evaluating')
    IoU_scores = []
    for i, data in enumerate(progress_bar):
        scale = data['scale']
        with torch.no_grad():
            output_list = model(data['img'].cuda().float())
        
        for j, output in enumerate(output_list):
            imgPath = test_set.getImagePath(i*opt.batch_size+j)
            scores, labels, boxes = output
            annot = data['annot'][j]
            annot = annot[annot[:, 4]!=-1]
            # print(scores.size(), labels.size(), boxes.size(), annot.size())
            if boxes.shape[0] == 0:
                if annot.size(0) == 0:
                    IoU_scores.append(1.0)
                else:
                    IoU_scores.append(0.0)
                continue
            if annot.size(0) == 0:
                IoU_scores.append(0.0)
            else:
                classes = set(annot[:, 4].tolist())
                cat = torch.cat([scores.view(-1, 1), labels.view(-1, 1).float(), boxes], dim=1)
                cat = cat[cat[:, 0]>=opt.cls_threshold]
                iou_score = []
                for c in classes:
                    box = cat[cat[:, 1]==c][:, 2:]
                    if box.size(0) == 0:
                        iou_score.append(0.0)
                        continue
                    tgt = annot[annot[:, 4]==c][:, :4]
                    iou_s = iou(box, tgt.cuda())
                    iou_score.append(iou_s.cpu().numpy())
                classes_pre = set(cat[:, 1].tolist())
                for c in classes_pre:
                    if c not in classes:
                        iou_score.append(0)
                # print(classes_pre, classes ,iou_score)
                IoU_scores.append(sum(iou_score)/len(iou_score))

            if writePIC:
                annot /= scale[j]
                boxes /= scale[j]
                # image_info = test_set.coco.loadImgs(test_set.image_ids[i*opt.batch_size+j])[0]
                # print(image_info['file_name'])
                # path = os.path.join(test_set.root_dir, 'images', test_set.set_name, image_info['file_name'])
                output_image = cv2.imread(os.path.join(opt.data_path, imgPath))
                # print(output_image.shape)
                for box_id in range(boxes.shape[0]):
                    pred_prob = float(scores[box_id])
                    if pred_prob < opt.cls_threshold:
                        break
                    pred_label = int(labels[box_id])
                    xmin, ymin, xmax, ymax = boxes[box_id, :]
                    color = colors[pred_label]
                    cv2.rectangle(output_image, (xmin, ymin), (xmax, ymax), color, 1)
                    text_size = cv2.getTextSize(test_set.index2label(pred_label) + ' : %.2f' % pred_prob, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]

                    cv2.rectangle(output_image, (xmin, ymin), (xmin + text_size[0] + 3, ymin + text_size[1] + 4), color, 3)
                    cv2.putText(
                        output_image, test_set.index2label(pred_label) + ' : %.2f' % pred_prob,
                        (xmin, ymin + text_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1,
                        (255, 255, 255), 1)
                for box_id in range(annot.size(0)):
                    xmin, ymin, xmax, ymax = annot[box_id, :4]
                    cv2.rectangle(output_image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 1)

                cv2.imwrite("{}/{}_prediction.jpg".format(opt.prediction_dir, imgPath.split('/')[-1][:-4]), output_image)
    print(sum(IoU_scores)/len(IoU_scores))

if __name__ == "__main__":
    opt = get_args_efficientdet()
    test(opt)