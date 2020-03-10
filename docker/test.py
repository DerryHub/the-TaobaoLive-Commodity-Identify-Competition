import os
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import TestImageDataset, TestVideoDataset, TestDataset
from utils import Resizer_Test, Normalizer_Test, collater_test
from efficientdet.efficientdet import EfficientDet
from arcface.resnet import ResNet
from arcface.googlenet import GoogLeNet
from arcface.inception_v4 import InceptionV4
from arcface.inceptionresnet_v2 import InceptionResNetV2
from config import get_args_efficientdet, get_args_arcface
from tqdm import tqdm
import json

def pre_efficient(dataset, model, opt_e):
    params = {"batch_size": opt_e.batch_size,
                    "shuffle": False,
                    "drop_last": False,
                    "collate_fn": collater_test,
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
                # vdo_id, img_id, score, vdo_index, img_index
    return vdo2img

def test(opt_a, opt_e):
    k = 10
    dir_list = ['test_dataset_part1', 'test_dataset_part2']

    dataset_img = TestImageDataset(
        root_dir=opt_e.data_path,
        dir_list=dir_list,
        transform=transforms.Compose([Normalizer_Test(), Resizer_Test()]))
    dataset_vdo = TestVideoDataset(
        root_dir=opt_e.data_path,
        dir_list=dir_list,
        transform=transforms.Compose([Normalizer_Test(), Resizer_Test()]))

    opt_e.num_classes = dataset_img.num_classes
    efficientdet = EfficientDet(opt_e)
    efficientdet.load_state_dict(torch.load(os.path.join(opt_e.saved_path, opt_e.network+'.pth')))
    efficientdet.cuda()
    efficientdet.set_is_training(False)
    efficientdet.eval()

    if opt_a.network == 'resnet':
        backbone = ResNet(opt_a)
        b_name = opt_a.network+'_'+opt_a.mode+'_{}'.format(opt_a.num_layers)
    elif opt_a.network == 'googlenet':
        backbone = GoogLeNet(opt_a)
        b_name = opt_a.network
    elif opt_a.network == 'inceptionv4':
        backbone = InceptionV4(opt_a)
        b_name = opt_a.network
    elif opt_a.network == 'inceptionresnetv2':
        backbone = InceptionResNetV2(opt_a)
        b_name = opt_a.network
    else:
        raise RuntimeError('Cannot Find the Model: {}'.format(opt_a.network))
        

    backbone.load_state_dict(torch.load(os.path.join(opt_a.saved_path, b_name+'.pth')))
    backbone.cuda()
    backbone.eval()
    
    print('predicting boxs...')
    imgs = pre_efficient(dataset_img, efficientdet, opt_e)
    vdos = pre_efficient(dataset_vdo, efficientdet, opt_e)
    
    dataset_det_img = TestDataset(opt_a.data_path, imgs, (opt_a.size, opt_a.size), mode='image')
    dataset_det_vdo = TestDataset(opt_a.data_path, vdos, (opt_a.size, opt_a.size), mode='video')

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

    vdo_dic = {}
    for i in range(len(vdo_IDs)):
        if vdo_IDs[i] not in vdo_dic:
            vdo_dic[vdo_IDs[i]] = {}
        if vdo_frames[i] not in vdo_dic[vdo_IDs[i]]:
            vdo_dic[vdo_IDs[i]][vdo_frames[i]] = []
        vdo_dic[vdo_IDs[i]][vdo_frames[i]].append(i)
    
    img_dic = {}
    for i in range(len(img_IDs)):
        if img_IDs[i] not in img_dic:
            img_dic[img_IDs[i]] = []
        img_dic[img_IDs[i]].append(i)
    
    result = {}
    print('testing...')
    for k in tqdm(vdo2img_d.keys()):
        max_pre = sorted(vdo2img_d[k].items(), key=lambda x:x[1][0], reverse=True)[0]
        vdo_id = vdo_IDs[max_pre[1][2]]
        img_id = img_IDs[max_pre[1][3]]
        frame_index = vdo_frames[max_pre[1][2]]
        vdo_index = vdo_dic[vdo_id][frame_index]
        img_index = img_dic[img_id]
        result[vdo_id] = {}
        result[vdo_id]['item_id'] = img_id
        result[vdo_id]['frame_index'] = frame_index
        result[vdo_id]['result'] = []
        vdo_f = np.zeros((0, opt_a.embedding_size))
        for index in vdo_index:
            vdo_f = np.append(vdo_f, vdo_features[index].reshape(1, opt_a.embedding_size), axis=0)
        img_f = np.zeros((0, opt_a.embedding_size))
        for index in img_index:
            img_f = np.append(img_f, img_features[index].reshape(1, opt_a.embedding_size), axis=0)
        cos = cosine_similarity(vdo_f, img_f)
        for i, index in enumerate(vdo_index):
            simis = [cos[i, j] for j in range(len(img_index))]
            simis_i = np.argmax(simis)
            if simis[simis_i] < 0:
                continue
            img_i = img_index[simis_i]
            d = {}
            d['img_name'] = img_frames[img_i]
            d['item_box'] = list(map(int, img_boxes[img_i].tolist()))
            d['frame_box'] = list(map(int, vdo_boxes[index].tolist()))
            result[vdo_id]['result'].append(d)

    with open('result.json', 'w') as f:
        json.dump(result, f)

if __name__ == "__main__":
    opt_e = get_args_efficientdet()
    opt_a = get_args_arcface()
    test(opt_a, opt_e)


