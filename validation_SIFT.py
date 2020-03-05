import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from config import get_args_arcface
from dataset import ValidationArcfaceDataset
from tqdm import tqdm
import json

def SIFT(img):
    b, c, h, w = img.size()
    img = img.view(b, c, -1)
    mean = torch.mean(img, axis=2).view(b, c, 1)
    features = (img>mean).long().view(b, -1)
    return features

def hamming_distance(features_1, features_2):
    assert features_1.shape == features_2.shape
    b, length = features_1.shape
    dis = np.zeros((0, b))
    print('Calculating Hamming distance...')
    for f in tqdm(features_1):
        f = f.reshape(1, length)
        dis = np.append(dis, 1-np.sum((f==features_2), axis=1).reshape(1, b)/np.float(length), axis=0)
    return dis

def evaluate(opt):
    size = 8
    dataset = ValidationArcfaceDataset(size=(size, size), root_dir='data/validation_instance/')
    opt.batch_size *= 10
    loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers)

    with open('data/instance2label.json', 'r') as f:
        ins2labDic = json.load(f)

    img_features = np.zeros((0, size**2*3))
    vdo_features = np.zeros((0, size**2*3))
    instances = []

    print('Predecting...')
    for d in tqdm(loader):
        img = d['img']
        vdo = d['vdo']
        instances += d['instance']
        
        # save_img(img[1].permute(1, 2, 0).numpy(), 'aaa')
        # save_img(vdo[1].permute(1, 2, 0).numpy(), 'bbb')
        # break
        img_f = SIFT(img).numpy()
        vdo_f = SIFT(vdo).numpy()

        img_features = np.append(img_features, img_f, axis=0)
        vdo_features = np.append(vdo_features, vdo_f, axis=0)

    dis = hamming_distance(vdo_features, img_features)

    acc = 0
    rates = []
    argmax = np.argsort(dis, axis=1)
    for i in tqdm(range(len(dis))):
        for j in argmax[i]:
            if ins2labDic[instances[i]] != ins2labDic[instances[j]]:
                continue
            if j == i:
                acc +=1
            rates.append(dis[i, j])
            break
    acc = acc/len(dis)

    print(sum(rates)/len(rates), min(rates), max(rates))
    print(acc)

def save_img(a, name):
    from PIL import Image
    print(a.shape)
    mi = np.min(a.reshape(-1))
    ma = np.max(a.reshape(-1))
    a = (a-mi)/(ma-mi)
    a = a*256
    img = Image.fromarray(a.astype(np.uint8))
    img.save('{}.jpg'.format(name))

if __name__ == "__main__":
    opt = get_args_arcface()
    evaluate(opt)
    # # def resize(src, new_size):
    # #     dst_w, dst_h = new_size # 目标图像宽高
    # #     src_h, src_w = src.shape[:2] # 源图像宽高
    # #     if src_h == dst_h and src_w == dst_w:
    # #         return src.copy()
    # #     scale_x = float(src_w) / dst_w # x缩放比例
    # #     scale_y = float(src_h) / dst_h # y缩放比例

    # #     # 遍历目标图像，插值
    # #     dst = np.zeros((dst_h, dst_w, 3))
    # #     for n in range(3): # 对channel循环
    # #         for dst_y in range(dst_h): # 对height循环
    # #             for dst_x in range(dst_w): # 对width循环
    # #                 # 目标在源上的坐标
    # #                 src_x = (dst_x + 0.5) * scale_x - 0.5
    # #                 src_y = (dst_y + 0.5) * scale_y - 0.5
    # #                 # 计算在源图上四个近邻点的位置
    # #                 src_x_0 = int(np.floor(src_x))
    # #                 src_y_0 = int(np.floor(src_y))
    # #                 src_x_1 = min(src_x_0 + 1, src_w - 1)
    # #                 src_y_1 = min(src_y_0 + 1, src_h - 1)

    # #                 # 双线性插值
    # #                 value0 = (src_x_1 - src_x) * src[src_y_0, src_x_0, n] + (src_x - src_x_0) * src[src_y_0, src_x_1, n]
    # #                 value1 = (src_x_1 - src_x) * src[src_y_1, src_x_0, n] + (src_x - src_x_0) * src[src_y_1, src_x_1, n]
    # #                 dst[dst_y, dst_x, n] = int((src_y_1 - src_y) * value0 + (src_y - src_y_0) * value1)
    # #     return dst
    # # # 
    # # a = [[[[1,2],[2,3]],[[4,5],[5,4]], [[6,5],[6,4]]],[[[1,2],[2,3]],[[4,5],[5,4]], [[6,5],[6,4]]]]
    # a = [[[0.1,0.2,0.3],[0.3,0.4,0.5]],[[0.1,0.2,0.3],[0.6,0.4,0.2]]]
    # a = np.array(a)
    # # print(resize(a, (1,1)))
    # # # a = torch.Tensor(a)
    # # # a = SIFT(a).numpy()
    # # # a = np.array([[0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0],
    # # #     [0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0]])
    # # # dis = hamming_distance(a, a)
    # # # print(np.argsort(dis, axis=1))
    # from PIL import Image
    # a = a*256
    # img = Image.fromarray(a.astype(np.uint8))
    # img.save('aaa.jpg')