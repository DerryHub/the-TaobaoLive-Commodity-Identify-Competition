import os
import torch
import numpy as np
from tqdm import tqdm
import json
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2

class EfficientdetDataset(Dataset):
    def __init__(self, root_dir='data', mode='train', transform=None):
        assert mode in ['train', 'validation']

        self.root_dir = root_dir
        self.transform = transform

        img_tat = mode + '_images'
        vdo_tat = mode + '_videos'

        with open(os.path.join(root_dir, img_tat+'_annotation.json'), 'r') as f:
            d_i = json.load(f)
        with open(os.path.join(root_dir, vdo_tat+'_annotation.json'), 'r') as f:
            d_v = json.load(f)

        labels = set(d_i['label2index'].keys()) | set(d_v['label2index'].keys())

        self.num_classes = len(labels)

        self.labelDic = {}
        self.labelDic['label2index'] = {}
        self.labelDic['index2label'] = {}

        for label in labels:
            self.labelDic['label2index'][label] = len(self.labelDic['label2index'])
            self.labelDic['index2label'][len(self.labelDic['index2label'])] = label

        l_i = d_i['annotations']
        l_v = d_v['annotations']

        self.images = []
        
        for d in tqdm(l_i):
            if len(d['annotations']) == 0:
                continue
            t = []
            t.append(os.path.join(img_tat, d['img_name']))
            for i in range(len(d['annotations'])):
                index = d['annotations'][i]['label']
                d['annotations'][i]['label'] = self.labelDic['label2index'][d_i['index2label'][str(index)]]
            t.append(d['annotations'])
            self.images.append(t)
            
        for d in tqdm(l_v):
            if len(d['annotations']) == 0:
                continue
            t = []
            t.append(os.path.join(vdo_tat, d['img_name']))
            for i in range(len(d['annotations'])):
                index = d['annotations'][i]['label']
                d['annotations'][i]['label'] = self.labelDic['label2index'][d_v['index2label'][str(index)]]
            t.append(d['annotations'])
            self.images.append(t)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        imgPath, annotationsList = self.images[index]
        img = cv2.imread(os.path.join(self.root_dir, imgPath))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255

        annotations = np.zeros((0, 5))
        for annotationDic in annotationsList:
            annotation = np.zeros((1, 5))
            annotation[0, :4] = annotationDic['box']
            annotation[0, 4] = annotationDic['label']
            annotations = np.append(annotations, annotation, axis=0)
        
        sample = {'img': img, 'annot': annotations}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def label2index(self, label):
        return self.labelDic['label2index'][label]

    def index2label(self, index):
        return self.labelDic['index2label'][index]

    def getImagePath(self, index):
        imgPath, annotationsList = self.images[index]
        return imgPath


class ArcfaceDataset(Dataset):
    def __init__(self, root_dir='data', mode='train', size=(128, 128), flip_x=0.5):
        assert mode in ['train', 'validation']

        self.root_dir = root_dir
        self.size = size
        self.flip_x = flip_x

        img_tat = mode + '_images'
        vdo_tat = mode + '_videos'

        with open(os.path.join(root_dir, img_tat+'_annotation.json'), 'r') as f:
            d_i = json.load(f)
        with open(os.path.join(root_dir, vdo_tat+'_annotation.json'), 'r') as f:
            d_v = json.load(f)

        l_i = d_i['annotations']
        l_v = d_v['annotations']

        self.images = []

        id_set = set([])

        self.clsDic = {}

        for d in l_i:
            for dd in d['annotations']:
                if dd['instance_id'] > 0:
                    t = []
                    t.append(os.path.join(img_tat, d['img_name']))
                    t.append(dd['box'])
                    t.append(dd['instance_id'])
                    self.images.append(t)
                    id_set.add(dd['instance_id'])

        for d in l_v:
            for dd in d['annotations']:
                if dd['instance_id'] > 0:
                    t = []
                    t.append(os.path.join(vdo_tat, d['img_name']))
                    t.append(dd['box'])
                    t.append(dd['instance_id'])
                    self.images.append(t)
                    id_set.add(dd['instance_id'])

        for i in id_set:
            self.clsDic[i] = len(self.clsDic)

        self.num_classes = len(self.clsDic)

        self.items = []
        img_t = tqdm(self.images)
        img_t.set_description_str('Loading Data')
        for imgPath, box, instance_id in img_t:
            img = cv2.imread(os.path.join(self.root_dir, imgPath))
            img = img[box[1]:box[3], box[0]:box[2], :]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255
            img = cv2.resize(img, self.size)
            label = torch.tensor(self.clsDic[instance_id])
            self.items.append([img, label])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        img, label = self.items[index]
        if np.random.rand() < self.flip_x:
            img = img[:, ::-1, :].copy()
        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1)
        transform = transforms.Normalize(
            mean=[0.55574415, 0.51230767, 0.51123354], 
            std=[0.21303795, 0.21604613, 0.21273348])
        img = transform(img)

        return {'img':img, 'label':label}
        
        

if __name__ == "__main__":
    dataset = ArcfaceDataset()
    print(dataset[0])
    # mean = np.zeros(3)
    # std = np.zeros(3)
    # for d in tqdm(dataset):
    #     img = d['img']
    #     for i in range(3):
    #         mean[i] += img[:, :, i].mean()
    #         std[i] += img[:, :, i].std()
    # mean = mean / len(dataset)
    # std = std / len(dataset)
    # print(mean, std)
    