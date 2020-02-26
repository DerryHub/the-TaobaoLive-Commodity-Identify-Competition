import os
import torch
import numpy as np
from tqdm import tqdm
import json
from torch.utils.data import Dataset, DataLoader
import cv2

class MyDataset(Dataset):
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



def collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]

    imgs = torch.from_numpy(np.stack(imgs, axis=0))

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    imgs = imgs.permute(0, 3, 1, 2)

    return {'img': imgs, 'annot': annot_padded, 'scale': scales}


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, common_size=512):
        image, annots = sample['img'], sample['annot']
        height, width, _ = image.shape
        if height > width:
            scale = common_size / height
            resized_height = common_size
            resized_width = int(width * scale)
        else:
            scale = common_size / width
            resized_height = int(height * scale)
            resized_width = common_size

        image = cv2.resize(image, (resized_width, resized_height))

        new_image = np.zeros((common_size, common_size, 3))
        new_image[0:resized_height, 0:resized_width] = image

        annots[:, :4] *= scale

        return {'img': torch.from_numpy(new_image), 'annot': torch.from_numpy(annots), 'scale': scale}


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):
        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            sample = {'img': image, 'annot': annots}

        return sample


class Normalizer(object):

    def __init__(self):
        self.mean = np.array([[[0.64364545, 0.60998588, 0.60550367]]])
        self.std = np.array([[[0.22700769, 0.23887326, 0.23833767]]])

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']

        return {'img': ((image.astype(np.float32) - self.mean) / self.std), 'annot': annots}

if __name__ == "__main__":
    from torchvision import transforms
    dataset = MyDataset()
    print(len(dataset))
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
    