import os
import torch
import numpy as np
from tqdm import tqdm
import json
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
import random

'''
    for efficientdet
'''

class EfficientdetDataset(Dataset):
    def __init__(self, root_dir='data', mode='train', imgORvdo='all', transform=None):
        assert mode in ['train', 'validation']
        assert imgORvdo in ['image', 'video', 'all']

        self.root_dir = root_dir
        self.transform = transform

        label_file = 'label.json'
        with open(os.path.join(root_dir, label_file), 'r') as f:
            self.labelDic = json.load(f)

        self.num_classes = len(self.labelDic['label2index'])

        if imgORvdo == 'image':
            tats = [mode + '_images']
        elif imgORvdo == 'video':
            tats = [mode + '_videos']
        else:
            tats = [mode + '_images', mode + '_videos']

        # img_tat = mode + '_images'
        # vdo_tat = mode + '_videos'

        ds = []
        for t in tats:
            with open(os.path.join(root_dir, t+'_annotation.json'), 'r') as f:
                ds.append(json.load(f))

        # with open(os.path.join(root_dir, img_tat+'_annotation.json'), 'r') as f:
        #     d_i = json.load(f)
        # with open(os.path.join(root_dir, vdo_tat+'_annotation.json'), 'r') as f:
        #     d_v = json.load(f)
        

        # l_i = d_i['annotations']
        # l_v = d_v['annotations']

        ls = [d['annotations'] for d in ds]

        self.images = []
        
        print('Loading {} {} data...'.format(mode, imgORvdo))
        for i, l in enumerate(ls):
            for d in l:
                if len(d['annotations']) == 0:
                    continue
                t = []
                t.append(os.path.join(tats[i], d['img_name']))
                t.append(d['annotations'])
                t.append(d['img_name'])
                self.images.append(t)
        # print(len(self.images))
        # self.images = self.images[:2000]
        # for d in l_i:
        #     if len(d['annotations']) == 0:
        #         continue
        #     t = []
        #     t.append(os.path.join(img_tat, d['img_name']))
        #     t.append(d['annotations'])
        #     self.images.append(t)
            
        # for d in l_v:
        #     if len(d['annotations']) == 0:
        #         continue
        #     t = []
        #     t.append(os.path.join(vdo_tat, d['img_name']))
        #     t.append(d['annotations'])
        #     self.images.append(t)
        print('Done')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        imgPath, annotationsList, imgName = self.images[index]
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
        return self.labelDic['index2label'][str(index)]

    def getImagePath(self, index):
        imgPath, annotationsList, imgName = self.images[index]
        return imgPath

    def getImageInfo(self, index):
        imgPath, annotationsList, imgName = self.images[index]
        imgID, frame = imgName[:-4].split('_')
        return imgPath, imgID, frame

'''
    for arcface
'''

class ArcfaceDataset(Dataset):
    def __init__(self, root_dir='data', mode='train', size=(112, 112), flip_x=0.5):
        assert mode in ['train']

        self.root_dir = root_dir
        self.size = size
        self.flip_x = flip_x

        img_tat = mode + '_images'
        vdo_tat = mode + '_videos'
        savePath = mode + '_instance'
        self.savePath = os.path.join(root_dir, savePath)

        with open(os.path.join(root_dir, img_tat+'_annotation.json'), 'r') as f:
            d_i = json.load(f)
        with open(os.path.join(root_dir, vdo_tat+'_annotation.json'), 'r') as f:
            d_v = json.load(f)

        l_i = d_i['annotations']
        l_v = d_v['annotations']

        images = []

        instance = {}
        s_i = set([])
        s_v = set([])

        self.clsDic = {}

        print('Loading data...')
        for d in l_i:
            for dd in d['annotations']:
                if dd['instance_id'] > 0:
                    s_i.add(dd['instance_id'])
                    if dd['instance_id'] not in instance:
                        instance[dd['instance_id']] = 1
                    else:
                        instance[dd['instance_id']] += 1
                    t = []
                    t.append(img_tat+str(dd['instance_id'])+d['img_name'])
                    t.append(dd['instance_id'])
                    images.append(t)

        for d in l_v:
            for dd in d['annotations']:
                if dd['instance_id'] > 0:
                    s_v.add(dd['instance_id'])
                    if dd['instance_id'] not in instance:
                        instance[dd['instance_id']] = 1
                    else:
                        instance[dd['instance_id']] += 1
                    t = []
                    t.append(vdo_tat+str(dd['instance_id'])+d['img_name'])
                    t.append(dd['instance_id'])
                    images.append(t)

        id_set = s_i & s_v

        self.images = []
        for l in images:
            if l[-1] in id_set and instance[l[-1]] > 10 and instance[l[-1]] < 20:
                self.images.append(l)

        for i in id_set:
            self.clsDic[i] = len(self.clsDic)

        self.num_classes = len(self.clsDic)
        print('Done')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        imgName, instance_id = self.images[index]
        img = np.load(os.path.join(self.savePath, imgName)[:-4]+'.npy')

        h, w, c = img.shape

        rh = random.randint(0, h-self.size[0])
        rw = random.randint(0, w-self.size[1])

        img = img[rh:self.size[0]+rh, rw:self.size[1]+rw, :]

        label = torch.tensor(self.clsDic[instance_id])

        if np.random.rand() < self.flip_x:
            img = img[:, ::-1, :].copy()
        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1)
        transform = transforms.Normalize(
            mean=[0.55574415, 0.51230767, 0.51123354], 
            std=[0.21303795, 0.21604613, 0.21273348])
        img = transform(img)

        return {'img':img, 'label':label}
        

'''
    for validation
'''

class ValidationArcfaceDataset(Dataset):
    def __init__(self, root_dir='data/validation_instance/'):
        self.root_dir = root_dir
        instances = os.listdir(root_dir)
        self.items = []
        for instance in instances:
            imgs = os.listdir(root_dir+instance)
            if len(imgs) < 2:
                continue
            l = []
            for img in imgs:
                if 'images' in img:
                    l.append(os.path.join(instance, img))
                    break
            if len(l) == 0:
                continue
            for img in imgs:
                if 'videos' in img:
                    l.append(os.path.join(instance, img))
                    break
            if len(l) < 2:
                continue
            l.append(instance)
            self.items.append(l)
        
    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        imgPath, vdoPath, instance = self.items[index]
        img = np.load(os.path.join(self.root_dir, imgPath))
        vdo = np.load(os.path.join(self.root_dir, vdoPath))

        transform = transforms.Normalize(
            mean=[0.55574415, 0.51230767, 0.51123354], 
            std=[0.21303795, 0.21604613, 0.21273348])

        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1)
        vdo = torch.from_numpy(vdo)
        vdo = vdo.permute(2, 0, 1)

        img = transform(img)
        vdo = transform(vdo)

        return {'img': img, 'vdo': vdo, 'instance':instance}


class ValidationDataset(Dataset):
    def __init__(self, root_dir, items, size):
        self.size = size
        self.root_dir = root_dir
        self.imgPath = None
        self.img = None
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        frame, imgID, imgPath, xmin, ymin, xmax, ymax = self.items[index]
        if imgPath != self.imgPath:
            self.imgPath = imgPath
            self.img = cv2.imread(os.path.join(self.root_dir, imgPath))
        det = self.img[ymin:ymax, xmin:xmax, :].copy()
        det = cv2.resize(det, self.size)
        det = cv2.cvtColor(det, cv2.COLOR_BGR2RGB)
        det = det.astype(np.float32) / 255

        transform = transforms.Normalize(
            mean=[0.55574415, 0.51230767, 0.51123354], 
            std=[0.21303795, 0.21604613, 0.21273348])
        
        det = torch.from_numpy(det)
        det = det.permute(2, 0, 1)

        det = transform(det)

        return {'img': det, 'imgID': imgID, 'frame': frame, 'box': np.array([xmin, ymin, xmax, ymax])}


if __name__ == "__main__":
    dataset = ArcfaceDataset()
    print(dataset[0]['img'].shape)
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
    
