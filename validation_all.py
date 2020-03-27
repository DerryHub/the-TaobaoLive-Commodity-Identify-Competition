import os
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import EfficientdetDataset, ValidationDataset
from utils import Resizer, Normalizer, collater, iou
from efficientdet.efficientdet import EfficientDet
from arcface.resnet import ResNet
from arcface.googlenet import GoogLeNet
from arcface.inception_v4 import InceptionV4
from arcface.inceptionresnet_v2 import InceptionResNetV2
from config import get_args_efficientdet, get_args_arcface
from tqdm import tqdm
import joblib
import json

def pre_efficient(dataset, model, opt_e, cls_k):
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
            scores, labels, all_labels, boxes = output
            if cls_k:
                argmax = np.argpartition(all_labels.cpu(), kth=-cls_k, axis=1)[:, -cls_k:]
            else:
                argmax = -np.ones([len(all_labels), 1])
            boxes /= scale[j]
            for box_id in range(boxes.shape[0]):
                pred_prob = float(scores[box_id])
                if pred_prob < opt_e.cls_threshold:
                    break
                xmin, ymin, xmax, ymax = boxes[box_id, :]
                l = [frame, imgID, imgPath, int(xmin), int(ymin), int(xmax), int(ymax), argmax[box_id].tolist()]
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
    classes = []

    progress_bar = tqdm(generator)
    for data in progress_bar:
        img = data['img'].cuda()
        imgID = data['imgID']
        frame = data['frame']
        box = data['box'].numpy()
        cs = [d.view(-1, 1) for d in data['classes']]
        cs = torch.cat(cs, dim=1).tolist()

        with torch.no_grad():
            features = model(img).cpu().numpy()
        IDs += imgID
        frames += frame
        classes += cs
        features_arr = np.append(features_arr, features, axis=0)
        boxes_arr = np.append(boxes_arr, box, axis=0)

    return features_arr, boxes_arr, IDs, frames, classes

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
            d = {}
            for am in argmax[i, :]:
                if cos[i, am] > 0.2:
                    if img_IDs[am] not in d:
                        d[img_IDs[am]] = [cos[i, am], cos[i, am], am]
                    else:
                        l = d[img_IDs[am]][:]
                        if cos[i, am] > l[1]:
                            l[1] = cos[i, am]
                            l[2] = am
                        l[0] += cos[i, am]
                        d[img_IDs[am]] = l
            if len(d) == 0:
                continue
            d = sorted(d.items(), key=lambda x:x[1][0], reverse=True)
            vdo2img.append([vdo_IDs[i+1000*index], d[0][0], d[0][1][0], i+1000*index, d[0][1][2]])

    return vdo2img

def validate(opt_a, opt_e):
    k = 10
    cls_k = 3
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

    efficientdet_image = EfficientDet(opt_e)
    efficientdet_video = EfficientDet(opt_e)
    
    efficientdet_image.load_state_dict(torch.load(os.path.join(opt_e.saved_path, opt_e.network+'_image'+'.pth')))
    efficientdet_video.load_state_dict(torch.load(os.path.join(opt_e.saved_path, opt_e.network+'_video'+'.pth')))
    
    efficientdet_image.cuda()
    efficientdet_video.cuda()

    efficientdet_image.set_is_training(False)
    efficientdet_video.set_is_training(False)

    efficientdet_image.eval()
    efficientdet_video.eval()


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
    imgs = pre_efficient(dataset_img, efficientdet_image, opt_e, cls_k)
    vdos = pre_efficient(dataset_vdo, efficientdet_video, opt_e, cls_k)
    
    dataset_det_img = ValidationDataset(opt_a.data_path, imgs, (opt_a.size, opt_a.size))
    dataset_det_vdo = ValidationDataset(opt_a.data_path, vdos, (opt_a.size, opt_a.size))

    print('creating features...')
    img_features, img_boxes, img_IDs, img_frames, img_classes = pre_bockbone(dataset_det_img, backbone, opt_a)
    vdo_features, vdo_boxes, vdo_IDs, vdo_frames, vdo_classes = pre_bockbone(dataset_det_vdo, backbone, opt_a)
    
    assert len(set([
        len(img_features), 
        len(img_boxes), 
        len(img_IDs), 
        len(img_frames), 
        len(img_classes)]))==1
    assert len(set([
        len(vdo_features), 
        len(vdo_boxes), 
        len(vdo_IDs), 
        len(vdo_frames), 
        len(vdo_classes)]))==1
    
    vdo2img = cal_cosine_similarity(vdo_features, img_features, vdo_IDs, img_IDs, k)

    vdo2img_d = {}
    print('merging videos...')
    for l in tqdm(vdo2img):
        if cls_k != 0:
            flag = False
            vdo_index = l[3]
            img_index = l[4]
            vdo_cls = vdo_classes[vdo_index]
            img_cls = img_classes[img_index]
            for v_c in vdo_cls:
                for v_i in img_cls:
                    if v_c == v_i:
                        flag = True
                        break
            if not flag:
                continue
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

    # with open(os.path.join(opt_a.data_path, 'validation_images_annotation.json'), 'r') as f:
    #     img_lst = json.load(f)['annotations']
    # with open(os.path.join(opt_a.data_path, 'validation_videos_annotation.json'), 'r') as f:
    #     vdo_lst = json.load(f)['annotations']

    
    # img_dic = {}
    # vdo_dic = {}
    # for d in img_lst:
    #     img_dic[d['img_name'][:-4]] = d['annotations']
    # for d in vdo_lst:
    #     vdo_dic[d['img_name'][:-4]] = d['annotations']

    # N_TP = [0]*3
    # N_P = [0]*3
    # N_GT = [0]*3

    # vdo_IDs_set = set(vdo_IDs)

    # N_GT[0] = len(vdo_IDs_set)
    # N_P[0] = len(vdo2img_d)

    # print('evaluating...')
    # for k in tqdm(vdo2img_d.keys()):
    #     vdo2img_d[k] = sorted(vdo2img_d[k].items(), key=lambda x:x[1][0], reverse=True)
    #     if k == vdo2img_d[k][0][0]:
    #         N_TP[0] += 1
    #         vdo_f = vdo_frames[vdo2img_d[k][0][1][2]]
    #         img_f = img_frames[vdo2img_d[k][0][1][3]]
    #         img_box_pre = torch.from_numpy(img_boxes[vdo2img_d[k][0][1][3]])
    #         img_box = []
    #         instances = []
    #         for annotation in img_dic[vdo2img_d[k][0][0]+ '_'+img_f]:
    #             instance = annotation['instance_id']
    #             if instance > 0:
    #                 instances.append(instance)
    #                 img_box.append(torch.from_numpy(np.array(annotation['box'])))
    #         if len(instances) > 0:
    #             N_GT[1] += 1
    #         if len(img_box) > 0:
    #             N_GT[2] += 1
    #         for annotation in vdo_dic[k+ '_'+vdo_f]:
    #             instance = annotation['instance_id']
    #             if instance in instances:
    #                 N_TP[1] += 1
    #                 for box in img_box:
    #                     IOU = iou(img_box_pre.unsqueeze(0), box.unsqueeze(0))
    #                     if IOU > 0.5:
    #                         N_TP[2] += 1
    #                         break
    #                 break
    
    # N_P[1] = N_TP[0]
    # N_P[2] = N_TP[1]
    
    # p = [N_TP[i]/N_P[i] for i in range(3)]
    # r = [N_TP[i]/N_GT[i] for i in range(3)]
    # s = [(2*p[i]*r[i])/(p[i]+r[i]) for i in range(3)]

    # for i in range(3):
    #     print('#######################################')
    #     print('N_P {}: {}\tN_GT {}: {}\tN_TP {}: {}'.format(i+1, N_P[i], i+1, N_GT[i], i+1, N_TP[i]))
    #     print('p {}: {}\tr {}: {}'.format(i+1, p[i], i+1, r[i]))
    #     print('s {}: {}'.format(i+1, s[i]))
    #     print()


if __name__ == "__main__":
    opt_e = get_args_efficientdet()
    opt_a = get_args_arcface()
    validate(opt_a, opt_e)
