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
    def __init__(self, maxLen, root_dir='data', PAD=0):
        with open(os.path.join(root_dir, 'vocab.json'), 'r') as f:
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
    for efficientdet
'''

class EfficientdetDataset(Dataset):
    def __init__(self, root_dir='data', mode='train', imgORvdo='all', transform=None, maxLen=64, PAD=0):
        assert mode in ['train', 'validation']
        assert imgORvdo in ['image', 'video', 'all']

        self.root_dir = root_dir
        self.transform = transform
        text2num = Text2Num(maxLen=maxLen, root_dir=root_dir, PAD=PAD)
        self.vocab_size = text2num.vocab_size
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

        self.textDic = {}
        ds = []
        for t in tats:
            with open(os.path.join(root_dir, t+'_annotation.json'), 'r') as f:
                ds.append(json.load(f))
            with open(os.path.join(root_dir, t+'_text.json'), 'r') as f:
                self.textDic[t] = json.load(f)

        for k in self.textDic.keys():
            for kk in self.textDic[k].keys():
                self.textDic[k][kk] = text2num(self.textDic[k][kk])

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
                t.append(tats[i])
                self.images.append(t)
        # print(len(self.images))
        # self.images = self.images[:1000]
        print('Done')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        imgPath, annotationsList, imgName, t = self.images[index]
        text_name = imgName.split('_')[0]
        text = self.textDic[t][text_name]
        text = torch.Tensor(text).long()
    
        img = cv2.imread(os.path.join(self.root_dir, imgPath))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255

        annotations = np.zeros((len(annotationsList), 6))
        for i, annotationDic in enumerate(annotationsList):
            annotation = np.zeros((1, 6))
            annotation[0, :4] = annotationDic['box']
            annotation[0, 4] = annotationDic['label']
            if annotationDic['instance_id'] > 0:
                annotation[0, 5] = 1
            else:
                annotation[0, 5] = 0
            annotations[i:i+1, :] = annotation
            # annotations = np.append(annotations, annotation, axis=0)
        
        sample = {'img': img, 'annot': annotations, 'text': text}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def label2index(self, label):
        return self.labelDic['label2index'][label]

    def index2label(self, index):
        return self.labelDic['index2label'][str(index)]

    def getImagePath(self, index):
        imgPath, annotationsList, imgName, t = self.images[index]
        return imgPath

    def getImageInfo(self, index):
        imgPath, annotationsList, imgName, t = self.images[index]
        imgID, frame = imgName[:-4].split('_')
        return imgPath, imgID, frame

'''
    for arcface
'''

class ArcfaceDataset(Dataset):
    def __init__(self, root_dir='data', mode='train', size=(112, 112), flip_x=0.5, maxLen=64, PAD=0):
        assert mode in ['train']

        self.root_dir = root_dir
        self.size = size
        self.flip_x = flip_x

        img_tat = mode + '_images'
        vdo_tat = mode + '_videos'
        savePath = mode + '_instance'
        self.savePath = os.path.join(root_dir, savePath)

        text2num = Text2Num(maxLen=maxLen, root_dir=root_dir, PAD=PAD)
        self.vocab_size = text2num.vocab_size

        with open(os.path.join(root_dir, img_tat+'_annotation.json'), 'r') as f:
            d_i = json.load(f)
        with open(os.path.join(root_dir, vdo_tat+'_annotation.json'), 'r') as f:
            d_v = json.load(f)

        with open(os.path.join(root_dir, img_tat+'_text.json'), 'r') as f:
            self.textDic_i = json.load(f)
        with open(os.path.join(root_dir, vdo_tat+'_text.json'), 'r') as f:
            self.textDic_v = json.load(f)
        
        for k in self.textDic_i.keys():
            self.textDic_i[k] = text2num(self.textDic_i[k])
        for k in self.textDic_v.keys():
            self.textDic_v[k] = text2num(self.textDic_v[k])

        l_i = d_i['annotations']
        l_v = d_v['annotations']

        self.images = []

        with open(os.path.join(root_dir, 'instanceID.json'), 'r') as f:
            self.clsDic = json.load(f)
        with open(os.path.join(root_dir, 'instance2label.json'), 'r') as f:
            self.instance2label = json.load(f)

        print('Loading data...')
        for d in l_i:
            for dd in d['annotations']:
                if dd['instance_id'] > 0 and str(dd['instance_id']) in self.clsDic.keys():
                    t = []
                    t.append(os.path.join(str(dd['instance_id']), img_tat+str(dd['instance_id'])+d['img_name']))
                    t.append(dd['instance_id'])
                    t.append(d['img_name'].split('_')[0])
                    t.append('image')
                    self.images.append(t)

        for d in l_v:
            for dd in d['annotations']:
                if dd['instance_id'] > 0 and str(dd['instance_id']) in self.clsDic.keys():
                    t = []
                    t.append(os.path.join(str(dd['instance_id']), vdo_tat+str(dd['instance_id'])+d['img_name']))
                    t.append(dd['instance_id'])
                    t.append(d['img_name'].split('_')[0])
                    t.append('video')
                    self.images.append(t)

        self.num_classes = len(self.clsDic)
        self.num_labels = len(set(self.instance2label.values()))
        print('Done')

        self.transform = transforms.Normalize(
            mean=[0.55574415, 0.51230767, 0.51123354], 
            std=[0.21303795, 0.21604613, 0.21273348])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        imgName, instance_id, textName, flag = self.images[index]
        img = np.load(os.path.join(self.savePath, imgName)[:-4]+'.npy')
        if flag == 'image':
            text = self.textDic_i[textName]
        elif flag == 'video':
            text = self.textDic_v[textName]
        text = torch.Tensor(text).long()

        h, w, c = img.shape

        rh = random.randint(0, h-self.size[0])
        rw = random.randint(0, w-self.size[1])

        img = img[rh:self.size[0]+rh, rw:self.size[1]+rw, :]

        instance = torch.tensor(self.clsDic[str(instance_id)])
        label = torch.tensor(self.instance2label[str(instance_id)])

        if np.random.rand() < self.flip_x:
            img = img[:, ::-1, :].copy()
        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1)
        
        img = self.transform(img)
        # r = torch.randn(3, self.size[0], self.size[1])
        # img = img + 0.1*r
        return {'img':img, 'instance':instance, 'label':label, 'text': text}

class TripletDataset(Dataset):
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

        with open(os.path.join(root_dir, 'instanceID.json'), 'r') as f:
            self.clsDic = json.load(f)
        
        with open(os.path.join(root_dir, 'instance2label.json'), 'r') as f:
            instance2label = json.load(f)

        l_i = d_i['annotations']
        l_v = d_v['annotations']

        self.images = []

        print('Loading data...')
        for d in l_i:
            for dd in d['annotations']:
                if dd['instance_id'] > 0 and str(dd['instance_id']) in self.clsDic.keys():
                    t = []
                    t.append(os.path.join(str(dd['instance_id']), img_tat+str(dd['instance_id'])+d['img_name']))
                    t.append(self.clsDic[str(dd['instance_id'])])
                    t.append(instance2label[str(dd['instance_id'])])
                    self.images.append(t)

        for d in l_v:
            for dd in d['annotations']:
                if dd['instance_id'] > 0 and str(dd['instance_id']) in self.clsDic.keys():
                    t = []
                    t.append(os.path.join(str(dd['instance_id']), vdo_tat+str(dd['instance_id'])+d['img_name']))
                    t.append(self.clsDic[str(dd['instance_id'])])
                    t.append(instance2label[str(dd['instance_id'])])
                    self.images.append(t)

        self.num_classes = len(self.clsDic)
        self.num_labels = len(set(instance2label.values()))

        self.cls_ins_dic = {}
        for i, l in enumerate(self.images):
            imgName, instance_id, label = l
            if label not in self.cls_ins_dic:
                self.cls_ins_dic[label] = {}
            if instance_id not in self.cls_ins_dic[label]:
                self.cls_ins_dic[label][instance_id] = []
            self.cls_ins_dic[label][instance_id].append(i)
        
        for k in self.cls_ins_dic.keys():
            if len(self.cls_ins_dic[k]) < 2:
                raise RuntimeError('size of self.cls_ins_dic[k] must be larger than 1')
        print('Done')
        self.transform = transforms.Normalize(
            mean=[0.55574415, 0.51230767, 0.51123354], 
            std=[0.21303795, 0.21604613, 0.21273348])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        imgName_q, instance_id_q, label_q = self.images[index]
        p_index = index
        while p_index == index:
            p_index = random.choice(self.cls_ins_dic[label_q][instance_id_q])
        instance_id_n = instance_id_q
        while instance_id_n == instance_id_q:
            instance_id_n = random.choice(list(self.cls_ins_dic[label_q].keys()))
        n_index = random.choice(self.cls_ins_dic[label_q][instance_id_n])
        imgName_p, instance_id_p, label_p = self.images[p_index]
        imgName_n, instance_id_n, label_n = self.images[n_index]

        assert len(set([label_q, label_p, label_n])) == 1
        assert len(set([instance_id_q, instance_id_p])) == 1

        instance_id_q = torch.tensor(instance_id_q)
        instance_id_p = torch.tensor(instance_id_p)
        instance_id_n = torch.tensor(instance_id_n)

        img_q = np.load(os.path.join(self.savePath, imgName_q)[:-4]+'.npy')
        img_p = np.load(os.path.join(self.savePath, imgName_p)[:-4]+'.npy')
        img_n = np.load(os.path.join(self.savePath, imgName_n)[:-4]+'.npy')

        hq, wq, cq = img_q.shape
        hp, wp, cp = img_p.shape
        hn, wn, cn = img_n.shape

        rh = random.randint(0, hq-self.size[0])
        rw = random.randint(0, wq-self.size[1])
        img_q = img_q[rh:self.size[0]+rh, rw:self.size[1]+rw, :]

        rh = random.randint(0, hp-self.size[0])
        rw = random.randint(0, wp-self.size[1])
        img_p = img_p[rh:self.size[0]+rh, rw:self.size[1]+rw, :]

        rh = random.randint(0, hn-self.size[0])
        rw = random.randint(0, wn-self.size[1])
        img_n = img_n[rh:self.size[0]+rh, rw:self.size[1]+rw, :]

        if np.random.rand() < self.flip_x:
            img_q = img_q[:, ::-1, :].copy()
        if np.random.rand() < self.flip_x:
            img_p = img_p[:, ::-1, :].copy()
        if np.random.rand() < self.flip_x:
            img_n = img_n[:, ::-1, :].copy()
        
        img_q = torch.from_numpy(img_q).permute(2, 0, 1)
        img_p = torch.from_numpy(img_p).permute(2, 0, 1)
        img_n = torch.from_numpy(img_n).permute(2, 0, 1)

        img_q = self.transform(img_q)
        img_p = self.transform(img_p)
        img_n = self.transform(img_n)

        return {
            'img_q':img_q, 
            'img_p':img_p, 
            'img_n':img_n, 
            'img_q_instance':instance_id_q,
            'img_p_instance':instance_id_p,
            'img_n_instance':instance_id_n,
        }


class HardTripletDataset(Dataset):
    def __init__(self, root_dir='data', mode='train', size=(112, 112), flip_x=0.5, n_samples=4):
        assert mode in ['train']

        self.root_dir = root_dir
        self.size = size
        self.flip_x = flip_x
        self.n_samples = n_samples

        img_tat = mode + '_images'
        vdo_tat = mode + '_videos'
        savePath = mode + '_instance'
        self.savePath = os.path.join(root_dir, savePath)

        with open(os.path.join(root_dir, img_tat+'_annotation.json'), 'r') as f:
            d_i = json.load(f)
        with open(os.path.join(root_dir, vdo_tat+'_annotation.json'), 'r') as f:
            d_v = json.load(f)

        with open(os.path.join(root_dir, 'instanceID.json'), 'r') as f:
            self.clsDic = json.load(f)

        l_i = d_i['annotations']
        l_v = d_v['annotations']

        self.samples = {}

        print('Loading data...')
        for d in l_i:
            for dd in d['annotations']:
                if dd['instance_id'] > 0 and str(dd['instance_id']) in self.clsDic.keys():
                    instance = self.clsDic[str(dd['instance_id'])]
                    if instance not in self.samples:
                        self.samples[instance] = []
                    self.samples[instance].append(
                        os.path.join(str(dd['instance_id']), img_tat+str(dd['instance_id'])+d['img_name']))
                    
        for d in l_v:
            for dd in d['annotations']:
                if dd['instance_id'] > 0 and str(dd['instance_id']) in self.clsDic.keys():
                    instance = self.clsDic[str(dd['instance_id'])]
                    if instance not in self.samples:
                        self.samples[instance] = []
                    self.samples[instance].append(
                        os.path.join(str(dd['instance_id']), vdo_tat+str(dd['instance_id'])+d['img_name']))

        self.num_classes = len(self.clsDic)
        
        for k in self.samples.keys():
            while len(self.samples[k]) < n_samples:
                self.samples[k] *= 2
            assert len(self.samples[k]) >= n_samples

        self.instances = list(self.samples.keys())
        print('Done')
        self.transform = transforms.Normalize(
            mean=[0.55574415, 0.51230767, 0.51123354], 
            std=[0.21303795, 0.21604613, 0.21273348])
    
    def __len__(self):
        return len(self.instances)
    
    def __getitem__(self, index):
        instance = self.instances[index]
        imgPaths = random.sample(self.samples[instance], self.n_samples)
        imgs = []
        instances = []
        for imgPath in imgPaths:
            img = np.load(os.path.join(self.savePath, imgPath)[:-4]+'.npy')

            h, w, c = img.shape

            rh = random.randint(0, h-self.size[0])
            rw = random.randint(0, w-self.size[1])

            img = img[rh:self.size[0]+rh, rw:self.size[1]+rw, :]

            instance_t = torch.tensor(instance)

            if np.random.rand() < self.flip_x:
                img = img[:, ::-1, :].copy()
            img = torch.from_numpy(img)
            img = img.permute(2, 0, 1)
            
            img = self.transform(img)
            
            imgs.append(img)
            instances.append(instance_t)
        imgs = torch.stack(imgs, dim=0)
        instances = torch.stack(instances, dim=0)

        return {'img': imgs, 'instance': instances}


'''
    for validation
'''

class ValidationArcfaceDataset(Dataset):
    def __init__(self, size=(112, 112), root_dir='data/validation_instance/'):
        self.root_dir = root_dir
        self.size = size
        instances = os.listdir(root_dir)
        self.items = []
        print('Loading Data...')
        for instance in tqdm(instances):
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
        self.length = len(self.items)
        print('Done')
        self.transform = transforms.Normalize(
            mean=[0.55574415, 0.51230767, 0.51123354], 
            std=[0.21303795, 0.21604613, 0.21273348])
    
    def __len__(self):
        return len(self.items) * 2

    def __getitem__(self, index):
        imgPath, vdoPath, instance = self.items[index%self.length]
        img = np.load(os.path.join(self.root_dir, imgPath))
        vdo = np.load(os.path.join(self.root_dir, vdoPath))
        
        hi, wi, ci = img.shape
        hv, wv, cv = vdo.shape

        rh = (hi-self.size[0])//2
        rw = (wi-self.size[1])//2
        img = img[rh:self.size[0]+rh, rw:self.size[1]+rw, :]

        rh = (hv-self.size[0])//2
        rw = (wv-self.size[1])//2
        vdo = vdo[rh:self.size[0]+rh, rw:self.size[1]+rw, :]

        if index >= self.length:
            img = img[:, ::-1, :].copy()
            vdo = vdo[:, ::-1, :].copy()

        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1)
        vdo = torch.from_numpy(vdo)
        vdo = vdo.permute(2, 0, 1)

        img = self.transform(img)
        vdo = self.transform(vdo)

        return {'img': img, 'vdo': vdo, 'instance':instance}


class ValidationDataset(Dataset):
    def __init__(self, root_dir, items, size):
        self.size = size
        self.root_dir = root_dir
        self.imgPath = None
        self.img = None
        self.items = items

        self.transform = transforms.Normalize(
            mean=[0.55574415, 0.51230767, 0.51123354], 
            std=[0.21303795, 0.21604613, 0.21273348])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        frame, imgID, imgPath, xmin, ymin, xmax, ymax, classes = self.items[index]
        if imgPath != self.imgPath:
            self.imgPath = imgPath
            self.img = cv2.imread(os.path.join(self.root_dir, imgPath))
        det = self.img[ymin:ymax, xmin:xmax, :].copy()
        det = cv2.resize(det, self.size)
        det = cv2.cvtColor(det, cv2.COLOR_BGR2RGB)
        det = det.astype(np.float32) / 255
        
        det = torch.from_numpy(det)
        det = det.permute(2, 0, 1)

        det = self.transform(det)
        # print(classes)
        return {
            'img': det, 
            'imgID': imgID, 
            'frame': frame, 
            'box': np.array([xmin, ymin, xmax, ymax]),
            'classes': classes}

'''
    for test
'''

class TestImageDataset(Dataset):
    def __init__(self, root_dir='data', dir_list=['validation_dataset_part1', 'validation_dataset_part2'], transform=None, maxLen=64, PAD=0):
        self.root_dir = root_dir
        self.transform = transform
        self.mode = 'image'

        label_file = 'label.json'
        with open(os.path.join(root_dir, label_file), 'r') as f:
            self.labelDic = json.load(f)

        self.num_classes = len(self.labelDic['label2index'])

        dirs = [os.path.join(root_dir, d) for d in dir_list]
        text2num = Text2Num(maxLen=maxLen, PAD=PAD)
        self.images = []
        self.ids = []
        self.frames = []
        self.textDic = {}
        
        for di in dirs:
            img_dir_list = os.listdir(os.path.join(di, 'image'))
            for img_dir in img_dir_list:
                img_names = os.listdir(os.path.join(di, 'image', img_dir))
                for img_name in img_names:
                    self.images.append(os.path.join(di, 'image', img_dir, img_name))
                    self.frames.append(img_name.split('.')[0])
                    self.ids.append(img_dir)
                textPath = os.path.join(di, 'image_text', img_dir+'.txt')
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
        text = torch.Tensor(text).long()
        sample = {'img': img, 'text': text}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def getImageInfo(self, index):
        imgPath = self.images[index]
        img_id = self.ids[index]
        frame = self.frames[index]
        return imgPath, img_id, frame


class TestVideoDataset(Dataset):
    def __init__(self, root_dir, transform=None, n=20, dir_list=['validation_dataset_part1', 'validation_dataset_part2'], maxLen=64, PAD=0):
        self.root_dir = root_dir
        self.transform = transform
        self.n = n
        self.mode = 'video'

        label_file = 'label.json'
        with open(os.path.join(root_dir, label_file), 'r') as f:
            self.labelDic = json.load(f)

        self.num_classes = len(self.labelDic['label2index'])
        text2num = Text2Num(maxLen=maxLen, PAD=PAD)
        dirs = [os.path.join(root_dir, d) for d in dir_list]

        gap = 400 // n
        self.frames_ids = [i*gap for i in range(n)]
        self.videos = []
        self.ids = []
        self.textDic = {}
        
        for di in dirs:
            vdo_names = os.listdir(os.path.join(di, 'video'))
            for vdo_name in vdo_names:
                self.videos.append(os.path.join(di, 'video', vdo_name))
                self.ids.append(vdo_name.split('.')[0])
                textPath = os.path.join(di, 'video_text', vdo_name.split('.')[0]+'.txt')
                with open(textPath, 'r') as f:
                    self.textDic[vdo_name.split('.')[0]] = text2num(f.readline())
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
        cap.release()

        vdo_id = self.ids[v_index]
        text = self.textDic[vdo_id]
        text = torch.Tensor(text).long()
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255
        
        sample = {'img': img, 'text': text}
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
        self.length = len(items)
        self.transform = transforms.Normalize(
            mean=[0.55574415, 0.51230767, 0.51123354], 
            std=[0.21303795, 0.21604613, 0.21273348])

    def __len__(self):
        return len(self.items) * 2

    def __getitem__(self, index):
        frame, imgID, imgPath, xmin, ymin, xmax, ymax, classes = self.items[index%self.length]
        if self.mode == 'image':
            img = cv2.imread(imgPath)
        else:
            cap = cv2.VideoCapture(imgPath)
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
            ret, img = cap.read()
            cap.release()
        det = img[ymin:ymax, xmin:xmax, :]
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
            'classes': classes}


if __name__ == "__main__":
    # from utils import collater_HardTriplet
    # from torch.utils.data import DataLoader
    # training_params = {"batch_size": 20,
    #                     "shuffle": True,
    #                     "drop_last": True,
    #                     "collate_fn": collater_HardTriplet,
    #                     "num_workers": 4}
    
#     from PIL import Image
    dataset = ArcfaceDataset()
    print(dataset[0])
    # loader = DataLoader(dataset, **training_params)
    # for data in loader:
    #     print(data['img'].size())
    #     break
    # print(len(dataset))
    # for d in tqdm(dataset):
    #     pass
    # img = dataset[100]['img']
    # mi = min(img.view(-1))
    # ma = max(img.view(-1))
    # img = (img-mi)/(ma-mi)
    # img = img*256
    # img = img.permute(1, 2, 0)
    # img = img.numpy()
    # img = Image.fromarray(img.astype(np.uint8))
    # img.save('aaa.jpg')
#     img = dataset[0]['vdo']
#     mi = min(img.view(-1))
#     ma = max(img.view(-1))
#     img = (img-mi)/(ma-mi)
#     img = img*256
#     img = img.permute(1, 2, 0)
#     img = img.numpy()
#     img = Image.fromarray(img.astype(np.uint8))
#     img.save('bbb.jpg')
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
    
