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
        all_ids = set([])

        self.images = []
        for l in images:
            if l[-1] in id_set and instance[l[-1]] > 10 and instance[l[-1]] < 20:
                self.images.append(l)
                all_ids.add(l[-1])
        
        for i in all_ids:
            self.clsDic[i] = len(self.clsDic)

        self.num_classes = len(self.clsDic)
        # self.images = self.images[:10000]
        print('Done')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        imgName, instance_id = self.images[index]
        img = np.load(os.path.join(self.savePath, imgName)[:-4]+'.npy')

        h, w, c = img.shape

        rh = random.randint(0, h-self.size[0])
        rw = random.randint(0, w-self.size[1])

        # if h == 128:
        #     rh = random.randint(0, h-self.size[0])
        #     rw = random.randint(0, w-self.size[0]*w//h)
        #     img = img[rh:self.size[0]+rh, rw:self.size[0]*w//h+rw, :]
        #     common_h = self.size[0]
        #     common_w = self.size[0]*w//h
        # elif w == 128:
        #     rh = random.randint(0, h-self.size[1]*h//w)
        #     rw = random.randint(0, w-self.size[1])
        #     img = img[rh:self.size[1]*h//w+rh, rw:self.size[1]+rw, :]
        #     common_h = self.size[1]*h//w
        #     common_w = self.size[1]
        # else:
        #     raise RuntimeError('shape error')


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

        # img_o = torch.zeros(3, self.size[0], self.size[1])
        # img_o[:, :common_h, :common_w] = img

        return {'img':img, 'label':label}

'''
    for classify
'''

class ClassifierDataset(Dataset):
    def __init__(self, size=(112, 112), root_dir='data', mode='train', flip_x=0.5):
        assert mode in ['train', 'validation']
        self.root_dir = root_dir
        self.size = size
        self.flip_x = flip_x
        self.mode = mode

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

        self.images = []

        self.clsDic = {}

        print('Loading data...')
        for d in l_i:
            for dd in d['annotations']:
                if dd['instance_id'] > 0:
                    t = []
                    if mode == 'train':
                        t.append(img_tat+str(dd['instance_id'])+d['img_name'])
                    else:
                        t.append(os.path.join(str(dd['instance_id']), img_tat+str(dd['instance_id'])+d['img_name']))
                    t.append(instance2label[str(dd['instance_id'])])
                    self.images.append(t)

        for d in l_v:
            for dd in d['annotations']:
                if dd['instance_id'] > 0:
                    t = []
                    if mode == 'train':
                        t.append(vdo_tat+str(dd['instance_id'])+d['img_name'])
                    else:
                        t.append(os.path.join(str(dd['instance_id']), vdo_tat+str(dd['instance_id'])+d['img_name']))
                    t.append(instance2label[str(dd['instance_id'])])
                    self.images.append(t)


        self.num_classes = len(instance2label)
        print('Done')
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        imgName, label = self.images[index]
        img = np.load(os.path.join(self.savePath, imgName)[:-4]+'.npy')

        if self.mode == 'train':
            h, w, c = img.shape

            rh = random.randint(0, h-self.size[0])
            rw = random.randint(0, w-self.size[1])

            img = img[rh:self.size[0]+rh, rw:self.size[1]+rw, :]

            if np.random.rand() < self.flip_x:
                img = img[:, ::-1, :].copy()

        label = torch.tensor(label)

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
    def __init__(self, size=(112, 112), root_dir='data/validation_instance/'):
        self.root_dir = root_dir
        self.size = size
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
        
    # def resize(self, src, new_size):
    #     dst_w, dst_h = new_size # 目标图像宽高
    #     src_h, src_w = src.shape[:2] # 源图像宽高
    #     if src_h == dst_h and src_w == dst_w:
    #         return src.copy()
    #     scale_x = float(src_w) / dst_w # x缩放比例
    #     scale_y = float(src_h) / dst_h # y缩放比例

    #     # 遍历目标图像，插值
    #     dst = np.zeros((dst_h, dst_w, 3))
    #     for n in range(3): # 对channel循环
    #         for dst_y in range(dst_h): # 对height循环
    #             for dst_x in range(dst_w): # 对width循环
    #                 # 目标在源上的坐标
    #                 src_x = (dst_x + 0.5) * scale_x - 0.5
    #                 src_y = (dst_y + 0.5) * scale_y - 0.5
    #                 # 计算在源图上四个近邻点的位置
    #                 src_x_0 = int(np.floor(src_x))
    #                 src_y_0 = int(np.floor(src_y))
    #                 src_x_1 = min(src_x_0 + 1, src_w - 1)
    #                 src_y_1 = min(src_y_0 + 1, src_h - 1)

    #                 # 双线性插值
    #                 value0 = (src_x_1 - src_x) * src[src_y_0, src_x_0, n] + (src_x - src_x_0) * src[src_y_0, src_x_1, n]
    #                 value1 = (src_x_1 - src_x) * src[src_y_1, src_x_0, n] + (src_x - src_x_0) * src[src_y_1, src_x_1, n]
    #                 dst[dst_y, dst_x, n] = float((src_y_1 - src_y) * value0 + (src_y - src_y_0) * value1)
    #     return dst
    
    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        imgPath, vdoPath, instance = self.items[index]
        img = np.load(os.path.join(self.root_dir, imgPath))
        vdo = np.load(os.path.join(self.root_dir, vdoPath))
        
        hi, wi, ci = img.shape
        hv, wv, cv = vdo.shape

        assert max(hi, wi) == 112
        assert max(hv, wv) == 112
        # assert (hi, wi, ci) == (hv, wv, cv)
        # if self.size != (hi, wi):
        #     img = self.resize(img, self.size)
        #     vdo = self.resize(vdo, self.size)

        transform = transforms.Normalize(
            mean=[0.55574415, 0.51230767, 0.51123354], 
            std=[0.21303795, 0.21604613, 0.21273348])

        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1)
        vdo = torch.from_numpy(vdo)
        vdo = vdo.permute(2, 0, 1)

        img = transform(img)
        vdo = transform(vdo)

        img_o = torch.zeros(3, self.size[0], self.size[1])
        img_o[:, :hi, :wi] = img

        vdo_o = torch.zeros(3, self.size[0], self.size[1])
        vdo_o[:, :hv, :wv] = vdo

        return {'img': img_o, 'vdo': vdo_o, 'instance':instance}


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
        frame, imgID, imgPath, xmin, ymin, xmax, ymax, classes = self.items[index]
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
        self.images = self.images[:1000]

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
        self.videos = self.videos[:100]

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


if __name__ == "__main__":
#     from PIL import Image
    dataset = TestImageDataset()
    # print(dataset[0])
    for d in tqdm(dataset):
        pass
#     img = dataset[0]['img']
#     mi = min(img.view(-1))
#     ma = max(img.view(-1))
#     img = (img-mi)/(ma-mi)
#     img = img*256
#     img = img.permute(1, 2, 0)
#     img = img.numpy()
#     img = Image.fromarray(img.astype(np.uint8))
#     img.save('aaa.jpg')
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
    
