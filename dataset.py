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

        ds = []
        for t in tats:
            with open(os.path.join(root_dir, t+'_annotation.json'), 'r') as f:
                ds.append(json.load(f))

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
        # self.images = self.images[:1000]
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

        self.images = []

        # instance = {}
        # s_i = set([])
        # s_v = set([])

        # self.clsDic = {}
        with open(os.path.join(root_dir, 'instanceID.json'), 'r') as f:
            self.clsDic = json.load(f)

        print('Loading data...')
        for d in l_i:
            for dd in d['annotations']:
                if dd['instance_id'] > 0 and str(dd['instance_id']) in self.clsDic.keys():
                    # s_i.add(dd['instance_id'])
                    # if dd['instance_id'] not in instance:
                    #     instance[dd['instance_id']] = 1
                    # else:
                    #     instance[dd['instance_id']] += 1
                    t = []
                    t.append(os.path.join(str(dd['instance_id']), img_tat+str(dd['instance_id'])+d['img_name']))
                    t.append(dd['instance_id'])
                    self.images.append(t)

        for d in l_v:
            for dd in d['annotations']:
                if dd['instance_id'] > 0 and str(dd['instance_id']) in self.clsDic.keys():
                    # s_v.add(dd['instance_id'])
                    # if dd['instance_id'] not in instance:
                    #     instance[dd['instance_id']] = 1
                    # else:
                    #     instance[dd['instance_id']] += 1
                    t = []
                    t.append(os.path.join(str(dd['instance_id']), vdo_tat+str(dd['instance_id'])+d['img_name']))
                    t.append(dd['instance_id'])
                    self.images.append(t)

        # id_set = s_i & s_v
        # all_ids = set([])
        # print(max(instance.values()))
        # self.images = []
        # for l in images:
        #     if l[-1] in id_set and instance[l[-1]] > 10 and instance[l[-1]] < 20:
        #         self.images.append(l)
        #         all_ids.add(l[-1])
        
        # all_ids = sorted(list(all_ids))

        # for i in all_ids:
        #     self.clsDic[i] = len(self.clsDic)

        self.num_classes = len(self.clsDic)

        # print(sorted(self.clsDic.items(), key=lambda x:x[1])[0])
        
        # self.images = self.images[:10000]
        print('Done')

        self.transform = transforms.Normalize(
            mean=[0.55574415, 0.51230767, 0.51123354], 
            std=[0.21303795, 0.21604613, 0.21273348])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        imgName, instance_id = self.images[index]
        img = np.load(os.path.join(self.savePath, imgName)[:-4]+'.npy')

        h, w, c = img.shape

        rh = random.randint(0, h-self.size[0])
        rw = random.randint(0, w-self.size[1])

        img = img[rh:self.size[0]+rh, rw:self.size[1]+rw, :]

        label = torch.tensor(self.clsDic[str(instance_id)])

        if np.random.rand() < self.flip_x:
            img = img[:, ::-1, :].copy()
        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1)
        
        img = self.transform(img)

        return {'img':img, 'label':label}

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
        
        with open(os.path.join(root_dir, 'instance2label.json'), 'r') as f:
            instance2label = json.load(f)

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
                    t.append(os.path.join(str(dd['instance_id']), img_tat+str(dd['instance_id'])+d['img_name']))
                    t.append(dd['instance_id'])
                    t.append(instance2label[str(dd['instance_id'])])
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
                    t.append(os.path.join(str(dd['instance_id']), vdo_tat+str(dd['instance_id'])+d['img_name']))
                    t.append(dd['instance_id'])
                    t.append(instance2label[str(dd['instance_id'])])
                    images.append(t)

        id_set = s_i & s_v
        all_ids = set([])

        self.images = []
        for l in images:
            if l[1] in id_set and instance[l[1]] > 10 and instance[l[1]] < 20:
                self.images.append(l)
                all_ids.add(l[-1])

        # self.images = self.images[:10000]

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

        return {'img_q':img_q, 'img_p':img_p, 'img_n':img_n}


class HTLDataset(Dataset):
    def __init__(self, root_dir='data', mode='train', size=(112, 112), flip_x=0.5, m=3, t=3):
        assert mode in ['train']

        self.root_dir = root_dir
        self.size = size
        self.flip_x = flip_x

        self.m = m
        self.t = t

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

        self.images = []

        with open(os.path.join(root_dir, 'instanceID.json'), 'r') as f:
            self.clsDic = json.load(f)

        print('Loading data...')
        for d in l_i:
            for dd in d['annotations']:
                if dd['instance_id'] > 0 and str(dd['instance_id']) in self.clsDic.keys():
                    t = []
                    t.append(os.path.join(str(dd['instance_id']), img_tat+str(dd['instance_id'])+d['img_name']))
                    t.append(dd['instance_id'])
                    self.images.append(t)

        for d in l_v:
            for dd in d['annotations']:
                if dd['instance_id'] > 0 and str(dd['instance_id']) in self.clsDic.keys():
                    t = []
                    t.append(os.path.join(str(dd['instance_id']), vdo_tat+str(dd['instance_id'])+d['img_name']))
                    t.append(dd['instance_id'])
                    self.images.append(t)

        self.num_classes = len(self.clsDic)

        self.instanceDic = {}
        for i, path, instance_id in enumerate(self.images):
            if instance_id not in self.instanceDic:
                self.instanceDic[instance_id] = []
            self.instanceDic[instance_id].append(i)

        print('Done')

        self.transform = transforms.Normalize(
            mean=[0.55574415, 0.51230767, 0.51123354], 
            std=[0.21303795, 0.21604613, 0.21273348])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index, minDic):
        imgName, instance_id = self.images[index]
        min_instances = minDic[instance_id][:self.m-1]
        
        # anchors = 
        img = np.load(os.path.join(self.savePath, imgName)[:-4]+'.npy')

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
        print('Done')
        self.transform = transforms.Normalize(
            mean=[0.55574415, 0.51230767, 0.51123354], 
            std=[0.21303795, 0.21604613, 0.21273348])
    
    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        imgPath, vdoPath, instance = self.items[index]
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
    def __init__(self, root_dir='data', dir_list=['validation_dataset_part1', 'validation_dataset_part2'], transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.mode = 'image'

        label_file = 'label.json'
        with open(os.path.join(root_dir, label_file), 'r') as f:
            self.labelDic = json.load(f)

        self.num_classes = len(self.labelDic['label2index'])

        dirs = [os.path.join(root_dir, d) for d in dir_list]

        self.images = []
        self.ids = []
        self.frames = []
        
        for di in dirs:
            img_dir_list = os.listdir(os.path.join(di, 'image'))
            for img_dir in img_dir_list:
                img_names = os.listdir(os.path.join(di, 'image', img_dir))
                for img_name in img_names:
                    self.images.append(os.path.join(di, 'image', img_dir, img_name))
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
    def __init__(self, root_dir='data', dir_list=['validation_dataset_part1', 'validation_dataset_part2'], transform=None, n=10):
        self.root_dir = root_dir
        self.transform = transform
        self.n = n
        self.mode = 'video'

        label_file = 'label.json'
        with open(os.path.join(root_dir, label_file), 'r') as f:
            self.labelDic = json.load(f)

        self.num_classes = len(self.labelDic['label2index'])

        dirs = [os.path.join(root_dir, d) for d in dir_list]

        gap = 400 // n
        self.frames_ids = [i*gap for i in range(n)]
        self.videos = []
        self.ids = []
        
        for di in dirs:
            vdo_names = os.listdir(os.path.join(di, 'video'))
            for vdo_name in vdo_names:
                self.videos.append(os.path.join(di, 'video', vdo_name))
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

        self.transform = transforms.Normalize(
            mean=[0.55574415, 0.51230767, 0.51123354], 
            std=[0.21303795, 0.21604613, 0.21273348])

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
            cap.release()
        det = img[ymin:ymax, xmin:xmax, :].copy()
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


# if __name__ == "__main__":
#     from PIL import Image
#     dataset = ArcfaceDataset()
    # print(dataset[0])
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
    
