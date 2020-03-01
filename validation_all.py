import os
import argparse
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import EfficientdetDataset, ValidationDataset
from utils import Resizer, Normalizer, collater
from efficientdet.efficientdet import EfficientDet
from arcface.backbone import Backbone
from config import get_args_efficientdet, get_args_arcface
from tqdm import tqdm


def validate(opt_a, opt_e):
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
    img_params = {"batch_size": opt_e.batch_size,
                    "shuffle": False,
                    "drop_last": False,
                    "collate_fn": collater,
                    "num_workers": opt_e.workers}

    vdo_params = {"batch_size": opt_e.batch_size,
                    "shuffle": False,
                    "drop_last": False,
                    "collate_fn": collater,
                    "num_workers": opt_e.workers}

    img_generator = DataLoader(dataset_img, **img_params)
    vdo_generator = DataLoader(dataset_vdo, **vdo_params)

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

    progress_bar_v = tqdm(vdo_generator)
    progress_bar_v.set_description_str(' Evaluating videos')
    vdos = []
    for i, data in enumerate(progress_bar_v):
        scale = data['scale']
        with torch.no_grad():
            output_list = efficientdet(data['img'].cuda().float())

        for j, output in enumerate(output_list):
            imgID = dataset_vdo.getImageID(i*opt_e.batch_size+j)
            imgPath = dataset_vdo.getImagePath(i*opt_e.batch_size+j)
            scores, labels, boxes = output
            boxes /= scale[j]
            for box_id in range(boxes.shape[0]):
                pred_prob = float(scores[box_id])
                if pred_prob < opt_e.cls_threshold:
                    break
                pred_label = int(labels[box_id])
                xmin, ymin, xmax, ymax = boxes[box_id, :]
                l = [imgID, imgPath, int(xmin), int(ymin), int(xmax), int(ymax)]
                vdos.append(l)

    progress_bar_i = tqdm(img_generator)
    progress_bar_i.set_description_str(' Evaluating images')
    imgs = []
    for i, data in enumerate(progress_bar_i):
        scale = data['scale']
        with torch.no_grad():
            output_list = efficientdet(data['img'].cuda().float())

        for j, output in enumerate(output_list):
            imgID = dataset_img.getImageID(i*opt_e.batch_size+j)
            imgPath = dataset_img.getImagePath(i*opt_e.batch_size+j)
            scores, labels, boxes = output
            boxes /= scale[j]
            for box_id in range(boxes.shape[0]):
                pred_prob = float(scores[box_id])
                if pred_prob < opt_e.cls_threshold:
                    break
                pred_label = int(labels[box_id])
                xmin, ymin, xmax, ymax = boxes[box_id, :]
                l = [imgID, imgPath, int(xmin), int(ymin), int(xmax), int(ymax)]
                imgs.append(l)
    
    

    dataset_det_img = ValidationDataset(opt_a.data_path, imgs, (opt_a.size, opt_a.size))
    dataset_det_vdo = ValidationDataset(opt_a.data_path, vdos, (opt_a.size, opt_a.size))

    img_det_params = {"batch_size": opt_a.batch_size,
                    "shuffle": False,
                    "drop_last": False,
                    "num_workers": opt_a.workers}

    vdo_det_params = {"batch_size": opt_a.batch_size,
                    "shuffle": False,
                    "drop_last": False,
                    "num_workers": opt_a.workers}

    img_det_generator = DataLoader(dataset_det_img, **img_det_params)
    vdo_det_generator = DataLoader(dataset_det_vdo, **vdo_det_params)

    img_features = np.zeros((0, 512))
    img_IDs = []
    vdo_features = np.zeros((0, 512))
    vdo_IDs = []

    progress_bar = tqdm(img_det_generator)
    progress_bar.set_description_str(' Evaluating images detection')
    for data in progress_bar:
        img = data['img'].cuda()
        imgID = data['imgID']
        with torch.no_grad():
            features = backbone(img).cpu().numpy()
        img_IDs += imgID
        img_features = np.append(img_features, features, axis=0)
    
    progress_bar = tqdm(vdo_det_generator)
    progress_bar.set_description_str(' Evaluating videos detection')
    for data in progress_bar:
        img = data['img'].cuda()
        imgID = data['imgID']
        with torch.no_grad():
            features = backbone(img).cpu().numpy()
        vdo_IDs += imgID
        vdo_features = np.append(vdo_features, features, axis=0)

    cos = cosine_similarity(vdo_features, img_features)
    argmax = np.argmax(cos, axis=1)
    
    N_TP = 0
    N_P = 0
    N_GT = 0
    for i in range(argmax.shape[0]):
        N_GT += 1
        if cos[i, argmax[i]] > 0:
            N_P += 1
            if vdo_IDs[i] == img_IDs[argmax[i]]:
                N_TP += 1
    
    print('N_P: {}\tN_GT: {}\tN_TP: {}'.format(N_P, N_GT, N_TP))
    p = N_TP/N_P
    r = N_TP/N_GT
    print(p, r)
    s_1 = (2*p*r)/(p+r)
    print(s_1)
        


if __name__ == "__main__":
    opt_e = get_args_efficientdet()
    opt_a = get_args_arcface()
    validate(opt_a, opt_e)
