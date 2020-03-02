import os
import argparse
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import EfficientdetDataset, ValidationDataset
from utils import Resizer, Normalizer, collater, iou
from efficientdet.efficientdet import EfficientDet
from arcface.backbone import Backbone
from config import get_args_efficientdet, get_args_arcface
from tqdm import tqdm
import datetime
import json

def pre_efficient(dataset, model, opt_e):
    params = {"batch_size": opt_e.batch_size,
                    "shuffle": False,
                    "drop_last": False,
                    "collate_fn": collater,
                    "num_workers": opt_e.workers}
    loader = DataLoader(dataset, **params)
    progress_bar = tqdm(loader)
    items = []
    for i, data in enumerate(progress_bar):
        scale = data['scale']
        with torch.no_grad():
            output_list = model(data['img'].cuda().float())
        for j, output in enumerate(output_list):
            imgPath, imgID, frame = dataset.getImageInfo(i*opt_e.batch_size+j)
            scores, labels, boxes = output
            boxes /= scale[j]
            for box_id in range(boxes.shape[0]):
                pred_prob = float(scores[box_id])
                if pred_prob < opt_e.cls_threshold:
                    break
                pred_label = int(labels[box_id])
                xmin, ymin, xmax, ymax = boxes[box_id, :]
                l = [frame, imgID, imgPath, int(xmin), int(ymin), int(xmax), int(ymax)]
                items.append(l)
    return items

def pre_bockbone(dataset, model, opt_a):
    params = {"batch_size": opt_a.batch_size,
                "shuffle": False,
                "drop_last": False,
                "num_workers": opt_a.workers}
    
    generator = DataLoader(dataset, **params)

    features_arr = np.zeros((0, opt_a.embedding_size))
    boxes_arr = np.zeros((0, 4))
    IDs = []
    frames = []

    progress_bar = tqdm(generator)
    for data in progress_bar:
        img = data['img'].cuda()
        imgID = data['imgID']
        frame = data['frame']
        box = data['box'].numpy()
        with torch.no_grad():
            features = model(img).cpu().numpy()
        IDs += imgID
        frames += frame
        features_arr = np.append(features_arr, features, axis=0)
        boxes_arr = np.append(boxes_arr, box, axis=0)

    return features_arr, boxes_arr, IDs, frames

def cal_cosine_similarity(vdo_features, img_features, vdo_IDs, img_IDs, k):
    vdo2img = []
    length = vdo_features.shape[0]
    print('calculating cosine similarity...')
    for index in tqdm(range(1+length//1000)):
        if index < length//1000:
            cos = cosine_similarity(vdo_features[1000*index:1000*(index+1)], img_features)
        else:
            cos = cosine_similarity(vdo_features[1000*index:], img_features)
        
        argmax = np.argpartition(cos, kth=-k, axis=1)[:, -k:]
        
        for i in range(argmax.shape[0]):
            if cos[i, argmax[i, 0]] > 0:
                d = {}
                for ind, am in enumerate(argmax[i, :]):
                    if img_IDs[am] not in d:
                        d[img_IDs[am]] = [cos[i, argmax[i, ind]], cos[i, argmax[i, ind]], am]
                    else:
                        l = d[img_IDs[am]]
                        if cos[i, argmax[i, ind]] > l[1]:
                            l[1] = cos[i, argmax[i, ind]]
                            l[2] = am
                        l[0] += cos[i, argmax[i, ind]]
                        d[img_IDs[am]] = l
                d = sorted(d.items(), key=lambda x:x[1][0], reverse=True)
                vdo2img.append([vdo_IDs[i+1000*index], d[0][0], d[0][1][0], i+1000*index, d[0][1][2]])
    return vdo2img


def validate(opt_a, opt_e):
    k = 10
    dataset_img = EfficientdetDataset(
        root_dir=opt_e.data_path, 
        mode='validation', 
        imgORvdo='image',
        transform=transforms.Compose([Normalizer(), Resizer()]))
    dataset_vdo = EfficientdetDataset(
        root_dir=opt_e.data_path, 
        mode='validation', 
        imgORvdo='video',
        transform=transforms.Compose([Normalizer(), Resizer()]))

    opt_e.batch_size *= 4

    opt_e.num_classes = dataset_img.num_classes
    efficientdet = EfficientDet(opt_e)
    efficientdet.load_state_dict(torch.load(os.path.join(opt_e.saved_path, opt_e.network+'.pth')))
    efficientdet.cuda()
    efficientdet.set_is_training(False)
    efficientdet.eval()

    backbone = Backbone(opt_a)
        
    b_name = 'backbone_'+opt_a.mode+'_{}'.format(opt_a.num_layers)

    backbone.load_state_dict(torch.load(os.path.join(opt_a.saved_path, b_name+'.pth')))
    backbone.cuda()
    backbone.eval()
    
    print('predicting boxs...')
    imgs = pre_efficient(dataset_img, efficientdet, opt_e)
    vdos = pre_efficient(dataset_vdo, efficientdet, opt_e)
    
    dataset_det_img = ValidationDataset(opt_a.data_path, imgs, (opt_a.size, opt_a.size))
    dataset_det_vdo = ValidationDataset(opt_a.data_path, vdos, (opt_a.size, opt_a.size))

    print('creating features...')
    img_features, img_boxes, img_IDs, img_frames = pre_bockbone(dataset_det_img, backbone, opt_a)
    vdo_features, vdo_boxes, vdo_IDs, vdo_frames = pre_bockbone(dataset_det_vdo, backbone, opt_a)

    vdo2img = cal_cosine_similarity(vdo_features, img_features, vdo_IDs, img_IDs, k)

    vdo2img_d = {}
    print('merging videos...')
    for l in tqdm(vdo2img):
        if l[0] not in vdo2img_d:
            vdo2img_d[l[0]] = {}
        if l[1] not in vdo2img_d[l[0]]:
            vdo2img_d[l[0]][l[1]] = [l[2], l[2], l[3], l[4]]
        else:
            lst = vdo2img_d[l[0]][l[1]]
            if l[2] > lst[1]:
                lst[1] = l[2]
                lst[2] = l[3]
                lst[3] = l[4]
            lst[0] += l[2]
            vdo2img_d[l[0]][l[1]] = lst
    
    with open(os.path.join(opt_a.data_path, 'validation_images_annotation.json'), 'r') as f:
        img_lst = json.load(f)['annotations']
    with open(os.path.join(opt_a.data_path, 'validation_videos_annotation.json'), 'r') as f:
        vdo_lst = json.load(f)['annotations']

    img_dic = {}
    vdo_dic = {}
    for d in img_lst:
        img_dic[d['img_name'][:-4]] = d['annotations']
    for d in vdo_lst:
        vdo_dic[d['img_name'][:-4]] = d['annotations']

    N_TP = [0]*3
    N_P = [0]*3
    N_GT = [0]*3

    vdo_IDs_set = set(vdo_IDs)

    N_GT[0] = len(vdo_IDs_set)
    N_P[0] = len(vdo2img_d)

    print('evaluating...')
    for k in tqdm(vdo2img_d.keys()):
        vdo2img_d[k] = sorted(vdo2img_d[k].items(), key=lambda x:x[1][0], reverse=True)
        if k == vdo2img_d[k][0][0]:
            N_TP[0] += 1
            vdo_f = vdo_frames[vdo2img_d[k][0][1][2]]
            img_f = img_frames[vdo2img_d[k][0][1][3]]
            img_box_pre = torch.from_numpy(img_boxes[vdo2img_d[k][0][1][3]])
            img_box = []
            instances = []
            for annotation in img_dic[vdo2img_d[k][0][0]+ '_'+img_f]:
                instance = annotation['instance_id']
                if instance > 0:
                    instances.append(instance)
                    img_box.append(torch.from_numpy(np.array(annotation['box'])))
            if len(instances) > 0:
                N_GT[1] += 1
            if len(img_box) > 0:
                N_GT[2] += 1
            for annotation in vdo_dic[k+ '_'+vdo_f]:
                instance = annotation['instance_id']
                if instance in instances:
                    N_TP[1] += 1
                    for box in img_box:
                        IOU = iou(img_box_pre.unsqueeze(0), box.unsqueeze(0))
                        if IOU > 0.5:
                            N_TP[2] += 1
                            break
                    break
    
    N_P[1] = N_TP[0]
    N_P[2] = N_TP[1]
    
    p = [N_TP[i]/N_P[i] for i in range(3)]
    r = [N_TP[i]/N_GT[i] for i in range(3)]
    s = [(2*p[i]*r[i])/(p[i]+r[i]) for i in range(3)]

    for i in range(3):
        print('#######################################')
        print('N_P {}: {}\tN_GT {}: {}\tN_TP {}: {}'.format(i+1, N_P[i], i+1, N_GT[i], i+1, N_TP[i]))
        print('p {}: {}\tr {}: {}'.format(i+1, p[i], i+1, r[i]))
        print('s {}: {}'.format(i+1, s[i]))
        print()


if __name__ == "__main__":
    opt_e = get_args_efficientdet()
    opt_a = get_args_arcface()
    validate(opt_a, opt_e)
