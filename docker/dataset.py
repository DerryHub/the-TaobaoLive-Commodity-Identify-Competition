import os
import torch
import numpy as np
from tqdm import tqdm
import json
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2
import random

'''
    for test
'''

class TestImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.mode = 'image'

        label_file = 'label.json'
        with open(label_file, 'r') as f:
            self.labelDic = json.load(f)

        self.num_classes = len(self.labelDic['label2index'])

        self.images = []
        self.ids = []
        self.frames = []
        
        img_dir_list = os.listdir(os.path.join(root_dir, 'image'))
        for img_dir in img_dir_list:
            img_names = os.listdir(os.path.join(root_dir, 'image', img_dir))
            for img_name in img_names:
                self.images.append(os.path.join(root_dir, 'image', img_dir, img_name))
                self.frames.append(img_name.split('.')[0])
                self.ids.append(img_dir)
        # self.images = self.images[:1000]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        imgPath = self.images[index]
        img = cv2.imread(imgPath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255
        
        sample = {'img': img}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def getImageInfo(self, index):
        imgPath = self.images[index]
        img_id = self.ids[index]
        frame = self.frames[index]
        return imgPath, img_id, frame


class TestVideoDataset(Dataset):
    def __init__(self, root_dir, transform=None, n=10):
        self.root_dir = root_dir
        self.transform = transform
        self.n = n
        self.mode = 'video'

        label_file = 'label.json'
        with open(label_file, 'r') as f:
            self.labelDic = json.load(f)

        self.num_classes = len(self.labelDic['label2index'])

        gap = 400 // n
        self.frames_ids = [i*gap for i in range(n)]
        self.videos = []
        self.ids = []
        
        vdo_names = os.listdir(os.path.join(root_dir, 'video'))
        for vdo_name in vdo_names:
            self.videos.append(os.path.join(root_dir, 'video', vdo_name))
            self.ids.append(vdo_name.split('.')[0])
        # self.videos = self.videos[:100]

    def __len__(self):
        return len(self.videos)*self.n

    def __getitem__(self, index):
        v_index = index // self.n
        f_index = self.frames_ids[index % self.n]
        vdo_name = self.videos[v_index]
        cap = cv2.VideoCapture(vdo_name)
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_index)
        ret, img = cap.read()
        # frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        # for i in range(int(frames)):
        #     ret, frame = cap.read()
        #     if i == f_index:
        #         img = frame
        #         break
        cap.release()
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255
        
        sample = {'img': img}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def getImageInfo(self, index):
        v_index = index // self.n
        frame = self.frames_ids[index % self.n]
        vdoPath = self.videos[v_index]
        vdo_id = self.ids[v_index]
        return vdoPath, vdo_id, str(frame)


class TestDataset(Dataset):
    def __init__(self, root_dir, items, size, mode):
        assert mode in ['image', 'video']
        self.mode = mode
        self.size = size
        self.root_dir = root_dir
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        frame, imgID, imgPath, xmin, ymin, xmax, ymax, classes = self.items[index]
        if self.mode == 'image':
            img = cv2.imread(imgPath)
        else:
            cap = cv2.VideoCapture(imgPath)
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
            ret, img = cap.read()
            # frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            # for i in range(int(frames)):
            #     ret, img = cap.read()
            #     if i == int(frame):
            #         break
            cap.release()
        det = img[ymin:ymax, xmin:xmax, :].copy()
        det = cv2.resize(det, self.size)
        det = cv2.cvtColor(det, cv2.COLOR_BGR2RGB)
        det = det.astype(np.float32) / 255

        transform = transforms.Normalize(
            mean=[0.55574415, 0.51230767, 0.51123354], 
            std=[0.21303795, 0.21604613, 0.21273348])
        
        det = torch.from_numpy(det)
        det = det.permute(2, 0, 1)

        det = transform(det)

        return {
            'img': det, 
            'imgID': imgID, 
            'frame': frame, 
            'box': np.array([xmin, ymin, xmax, ymax]),
            'classes': classes}