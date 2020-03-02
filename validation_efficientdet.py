import os
import argparse
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import EfficientdetDataset
from utils import Resizer, Normalizer, collater, iou
from efficientdet.config import colors
import cv2
import shutil
from efficientdet.efficientdet import EfficientDet
from config import get_args_efficientdet
from tqdm import tqdm


writePIC = False
calIOU = False
calPR = True
    

def test(opt):
    opt.resume = True
    test_set = EfficientdetDataset(opt.data_path, mode='validation', transform=transforms.Compose([Normalizer(), Resizer()]))
    opt.num_classes = test_set.num_classes
    opt.batch_size = opt.batch_size*4
    test_params = {"batch_size": opt.batch_size,
                   "shuffle": False,
                   "drop_last": False,
                   "collate_fn": collater,
                   "num_workers": opt.workers}
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
    N_TP = 0
    N_P = 0
    N_GT = 0
    N_TP_iou = 0
    for i, data in enumerate(progress_bar):
        scale = data['scale']
        with torch.no_grad():
            output_list = model(data['img'].cuda().float())
        
        for j, output in enumerate(output_list):
            imgPath = test_set.getImagePath(i*opt.batch_size+j)
            scores, labels, boxes = output
            annot = data['annot'][j]
            annot = annot[annot[:, 4]!=-1]

            cat = torch.cat([scores.view(-1, 1), labels.view(-1, 1).float(), boxes], dim=1)
            cat = cat[cat[:, 0]>=opt.cls_threshold]
            # print(scores.size(), labels.size(), boxes.size(), annot.size())
            if calIOU:
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
            
            if calPR:
                N_P += cat.size(0)
                N_GT += annot.size(0)
                for pre in cat:
                    for gt in annot:
                        s = iou(pre[2:].unsqueeze(0), gt[:4].unsqueeze(0))
                        if s > 0.5:
                            N_TP_iou += 1
                            if pre[1] == gt[-1]:
                                N_TP += 1
            

            if writePIC:
                annot_labels = annot[:, 4].clone()
                annot /= scale[j]
                boxes /= scale[j]
                output_image = cv2.imread(os.path.join(opt.data_path, imgPath))
                for box_id in range(boxes.shape[0]):
                    pred_prob = float(scores[box_id])
                    if pred_prob < opt.cls_threshold:
                        break
                    pred_label = int(labels[box_id])
                    xmin, ymin, xmax, ymax = boxes[box_id, :]
                    color = colors[pred_label]
                    cv2.rectangle(output_image, (xmin, ymin), (xmax, ymax), color, 2)
                    text_size = cv2.getTextSize('p: {}'.format(pred_label) + ' : %.2f' % pred_prob, cv2.FONT_HERSHEY_PLAIN, 3, 2)[0]

                    cv2.rectangle(output_image, (xmin, ymin), (xmin + text_size[0] + 3, ymin + text_size[1] + 4), color, 1)
                    cv2.putText(
                        output_image, 'p: {}'.format(pred_label) + ' : %.2f' % pred_prob,
                        (xmin, ymin + text_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1,
                        color, 2)
                for box_id in range(annot.size(0)):
                    true_label = int(annot_labels[box_id])
                    xmin, ymin, xmax, ymax = annot[box_id, :4]
                    cv2.rectangle(output_image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
                    text_size = cv2.getTextSize('g: {}'.format(true_label), cv2.FONT_HERSHEY_PLAIN, 3, 2)[0]
                    cv2.rectangle(output_image, (xmin, ymax), (xmin + text_size[0] + 3, ymax - text_size[1] + 4), (255, 0, 0), 1)
                    cv2.putText(
                        output_image, 'g: {}'.format(true_label),
                        (xmin, ymax - text_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1,
                        (255, 0, 0), 2)
                cv2.imwrite("{}/{}_prediction.jpg".format(opt.prediction_dir, imgPath.split('/')[-1][:-4]), output_image)
    
    if calIOU:
        print(sum(IoU_scores)/len(IoU_scores))
    if calPR:
        print('N_P: {}\tN_GT: {}\tN_TP_iou: {}\tN_TP: {}'.format(N_P, N_GT, N_TP_iou, N_TP))
        print('精确率: {}\t召回率: {}\t分类准确率: {}'.format(N_TP/N_P, N_TP/N_GT, N_TP/N_TP_iou))

if __name__ == "__main__":
    opt = get_args_efficientdet()
    test(opt)