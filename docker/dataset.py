import os
import torch
import numpy as np
from tqdm import tqdm
import json
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2
import random
import jieba

'''
    for text
'''

class Text2Num:
    def __init__(self, maxLen, PAD=0):
        with open('vocab.json', 'r') as f:
            self.vocab = json.load(f)
        self.PAD = PAD
        self.maxLen = maxLen
        self.vocab_size = len(self.vocab)

    def __call__(self, text):
        words = jieba.cut(text, cut_all=False, HMM=True)
        l = []
        for w in words:
            if w.strip() in self.vocab:
                l.append(self.vocab[w.strip()])
        if len(l) > self.maxLen:
            l = l[:self.maxLen]
        elif len(l) < self.maxLen:
            l += [self.PAD]*(self.maxLen-len(l))
        
        assert len(l) == self.maxLen

        return l

'''
    for test
'''

# class TestImageDataset(Dataset):
#     def __init__(self, root_dir='data', dir_list=['validation_dataset_part1', 'validation_dataset_part2'], transform=None, maxLen=64, PAD=0):
#         self.root_dir = root_dir
#         self.transform = transform
#         self.mode = 'image'

#         label_file = 'label.json'
#         with open(label_file, 'r') as f:
#             self.labelDic = json.load(f)

#         self.num_classes = len(self.labelDic['label2index'])

#         dirs = [os.path.join(root_dir, d) for d in dir_list]
#         text2num = Text2Num(maxLen=maxLen, PAD=PAD)
#         self.vocab_size = text2num.vocab_size
#         self.images = []
#         self.ids = []
#         self.frames = []
#         self.textDic = {}
        
#         for di in dirs:
#             img_dir_list = os.listdir(os.path.join(di, 'image'))
#             for img_dir in img_dir_list:
#                 img_names = os.listdir(os.path.join(di, 'image', img_dir))
#                 for img_name in img_names:
#                     self.images.append(os.path.join(di, 'image', img_dir, img_name))
#                     self.frames.append(img_name.split('.')[0])
#                     self.ids.append(img_dir)
#                 textPath = os.path.join(di, 'image_text', img_dir+'.txt')
#                 with open(textPath, 'r') as f:
#                     self.textDic[img_dir] = text2num(f.readline())
    
#         # self.images = self.images[:100]

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, index):
#         imgPath = self.images[index]
#         img = cv2.imread(imgPath)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = img.astype(np.float32) / 255
#         img_id = self.ids[index]
#         text = self.textDic[img_id]
#         text = torch.Tensor(text).long()
#         sample = {'img': img, 'text': text}
#         if self.transform:
#             sample = self.transform(sample)
#         return sample

#     def getImageInfo(self, index):
#         imgPath = self.images[index]
#         img_id = self.ids[index]
#         frame = self.frames[index]
#         return imgPath, img_id, frame

class TestImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, maxLen=64, PAD=0):
        self.root_dir = root_dir
        self.transform = transform
        self.mode = 'image'

        label_file = 'label.json'
        with open(label_file, 'r') as f:
            self.labelDic = json.load(f)

        self.num_classes = len(self.labelDic['label2index'])

        text2num = Text2Num(maxLen=maxLen, PAD=PAD)
        self.vocab_size = text2num.vocab_size
        self.images = []
        self.ids = []
        self.frames = []
        self.textDic = {}
        
        img_dir_list = os.listdir(os.path.join(root_dir, 'image'))
        for img_dir in img_dir_list:
            img_names = os.listdir(os.path.join(root_dir, 'image', img_dir))
            for img_name in img_names:
                self.images.append(os.path.join(root_dir, 'image', img_dir, img_name))
                self.frames.append(img_name.split('.')[0])
                self.ids.append(img_dir)
            textPath = os.path.join(root_dir, 'image_text', img_dir+'.txt')
            with open(textPath, 'r') as f:
                self.textDic[img_dir] = text2num(f.readline())
        

        # self.images = self.images[:1000]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        imgPath = self.images[index]
        img = cv2.imread(imgPath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255
        img_id = self.ids[index]
        text = self.textDic[img_id]
        text = torch.tensor(text).long()
        sample = {'img': img, 'text': text}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def getImageInfo(self, index):
        imgPath = self.images[index]
        img_id = self.ids[index]
        frame = self.frames[index]
        return imgPath, img_id, frame

# class TestVideoDataset(Dataset):
#     def __init__(self, root_dir, transform=None, n=20, dir_list=['validation_dataset_part1', 'validation_dataset_part2'], maxLen=64, PAD=0):
#         self.root_dir = root_dir
#         self.transform = transform
#         self.n = n
#         self.mode = 'video'

#         label_file = 'label.json'
#         with open(label_file, 'r') as f:
#             self.labelDic = json.load(f)

#         self.num_classes = len(self.labelDic['label2index'])
#         text2num = Text2Num(maxLen=maxLen, PAD=PAD)
#         self.vocab_size = text2num.vocab_size
#         dirs = [os.path.join(root_dir, d) for d in dir_list]

#         # gap = 400 // n
#         # self.frames_ids = [i*gap for i in range(n)]
#         self.videos = []
#         self.ids = []
#         self.textDic = {}
        
#         for di in dirs:
#             vdo_names = os.listdir(os.path.join(di, 'video'))
#             for vdo_name in vdo_names:
#                 self.videos.append(os.path.join(di, 'video', vdo_name))
#                 self.ids.append(vdo_name.split('.')[0])
#                 textPath = os.path.join(di, 'video_text', vdo_name.split('.')[0]+'.txt')
#                 with open(textPath, 'r') as f:
#                     self.textDic[vdo_name.split('.')[0]] = text2num(f.readline())
                    
#         # self.videos = self.videos[:10]

#     def __len__(self):
#         return len(self.videos)*self.n

#     def __getitem__(self, index):
#         v_index = index // self.n
#         # f_index = self.frames_ids[index % self.n]
#         vdo_name = self.videos[v_index]
#         cap = cv2.VideoCapture(vdo_name)
#         frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
#         f_index = int((frames // self.n) * (index % self.n))
#         cap.set(cv2.CAP_PROP_POS_FRAMES, f_index)
#         ret, img = cap.read()
#         cap.release()

#         vdo_id = self.ids[v_index]
#         text = self.textDic[vdo_id]
#         text = torch.Tensor(text).long()
        
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = img.astype(np.float32) / 255
        
#         sample = {'img': img, 'text': text}
#         if self.transform:
#             sample = self.transform(sample)
#         return sample

#     def getImageInfo(self, index):
#         v_index = index // self.n
#         # frame = self.frames_ids[index % self.n]
#         vdoPath = self.videos[v_index]
#         cap = cv2.VideoCapture(vdoPath)
#         frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
#         frame = int((frames // self.n) * (index % self.n))
#         cap.release()
#         vdo_id = self.ids[v_index]
#         return vdoPath, vdo_id, str(frame)


class TestVideoDataset(Dataset):
    def __init__(self, root_dir, transform=None, n=20, maxLen=64, PAD=0):
        self.root_dir = root_dir
        self.transform = transform
        self.n = n
        self.mode = 'video'

        label_file = 'label.json'
        with open(label_file, 'r') as f:
            self.labelDic = json.load(f)

        self.num_classes = len(self.labelDic['label2index'])
        text2num = Text2Num(maxLen=maxLen, PAD=PAD)
        self.vocab_size = text2num.vocab_size   
        # gap = 400 // n
        # self.frames_ids = [i*gap for i in range(n)]
        self.videos = []
        self.ids = []
        self.textDic = {}
        
        vdo_names = os.listdir(os.path.join(root_dir, 'video'))
        for vdo_name in vdo_names:
            self.videos.append(os.path.join(root_dir, 'video', vdo_name))
            self.ids.append(vdo_name.split('.')[0])
            textPath = os.path.join(root_dir, 'video_text', vdo_name.split('.')[0]+'.txt')
            with open(textPath, 'r') as f:
                self.textDic[vdo_name.split('.')[0]] = text2num(f.readline())
        # self.videos = self.videos[:100]

    def __len__(self):
        return len(self.videos)*self.n

    def __getitem__(self, index):
        v_index = index // self.n
        # f_index = self.frames_ids[index % self.n]
        vdo_name = self.videos[v_index]
        cap = cv2.VideoCapture(vdo_name)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        f_index = int((frames // self.n) * (index % self.n))
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_index)
        ret, img = cap.read()
        cap.release()

        vdo_id = self.ids[v_index]
        text = self.textDic[vdo_id]
        text = torch.tensor(text).long()
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255
        
        sample = {'img': img, 'text': text}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def getImageInfo(self, index):
        v_index = index // self.n
        # frame = self.frames_ids[index % self.n]
        vdoPath = self.videos[v_index]
        cap = cv2.VideoCapture(vdoPath)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        frame = int((frames // self.n) * (index % self.n))
        cap.release()
        vdo_id = self.ids[v_index]
        return vdoPath, vdo_id, str(frame)


class TestDataset(Dataset):
    def __init__(self, root_dir, items, size, mode):
        assert mode in ['image', 'video']
        self.mode = mode
        self.size = size
        self.root_dir = root_dir
        self.items = items
        self.length = len(items)
        self.transform = transforms.Normalize(
            mean=[0.55574415, 0.51230767, 0.51123354], 
            std=[0.21303795, 0.21604613, 0.21273348])

    def __len__(self):
        return len(self.items) * 2

    def __getitem__(self, index):
        frame, imgID, imgPath, xmin, ymin, xmax, ymax, classes, text = self.items[index%self.length]
        if self.mode == 'image':
            img = cv2.imread(imgPath)
        else:
            cap = cv2.VideoCapture(imgPath)
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
            ret, img = cap.read()
            cap.release()
        
        h, w, c = img.shape
        dy = int((ymax-ymin)*0.05)
        dx = int((xmax-xmin)*0.05)
        ymin_n = max(0, ymin-dy)
        ymax_n = min(h, ymax+dy)
        xmin_n = max(0, xmin-dx)
        xmax_n = min(w, xmax+dx)

        det = img[ymin_n:ymax_n, xmin_n:xmax_n, :]
        if index >= self.length:
            det = det[:, ::-1, :].copy()
        det = cv2.resize(det, self.size)

        det = cv2.cvtColor(det, cv2.COLOR_BGR2RGB)
        det = det.astype(np.float32) / 255
        det = torch.from_numpy(det)
        det = det.permute(2, 0, 1)
        det = self.transform(det)

        return {
            'img': det,
            'imgID': imgID, 
            'frame': frame, 
            'box': np.array([xmin, ymin, xmax, ymax]),
            'classes': classes,
            'text': text}