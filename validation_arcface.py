import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from arcface.backbone import Backbone
from arcface.head import Arcface
from config import get_args_arcface
from dataset import ValidationArcfaceDataset, ArcfaceDataset
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import json
    

def kmeans_classifer(opt, vdo_features, img_features, instances, ins2labDic):
    print('Classifying with k-means...')
    kmeans = joblib.load(os.path.join(opt.saved_path, 'kmeans.m'))
    vdo_cls = kmeans.predict(vdo_features)
    img_cls = kmeans.predict(img_features)

    vdo_cls = np.array(vdo_cls)
    img_cls = np.array(img_cls)

    acc = 0
    miss = 0
    rates = []
    for i, vc in enumerate(tqdm(vdo_cls)):
        l = []
        arg = np.argwhere(img_cls==vc).reshape(-1)
        for j in arg:
            if ins2labDic[instances[i]] == ins2labDic[instances[j]]:
                similarity = vdo_features[i]@img_features[j].T
                l.append([j, similarity])
        if len(l) == 0:
            miss += 1
            continue
        m = max(l, key=lambda x: x[1])
        if m[0] == i:
            acc += 1
        rates.append(m[1])
    print(miss/len(vdo_cls))
    return rates, acc/len(vdo_cls)


def cal_cosine_similarity(vdo_features, img_features, instances, ins2labDic):
    print('Calculating cosine similarity...')
    cos = cosine_similarity(vdo_features, img_features)
    argmax = np.argsort(-cos, axis=1)
    acc = 0
    rates = []
    for i in tqdm(range(len(cos))):
        for j in argmax[i]:
            if ins2labDic[instances[i]] != ins2labDic[instances[j]]:
                continue
            if j == i:
                acc +=1
            rates.append(cos[i, j])
            break
    return rates, acc/len(cos)
    

def evaluate(opt):
    dataset = ValidationArcfaceDataset(root_dir='data/validation_instance/')
    opt.batch_size *= 4
    loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers)

    model = Backbone(opt)
    
    b_name = 'backbone_'+opt.mode+'_{}'.format(opt.num_layers)

    model.load_state_dict(torch.load(os.path.join(opt.saved_path, b_name+'.pth')))
    model.cuda()
    model.eval()

    img_features = np.zeros((0, opt.embedding_size))
    vdo_features = np.zeros((0, opt.embedding_size))
    instances = []

    print('Predecting...')
    for d in tqdm(loader):
        img = d['img']
        vdo = d['vdo']
        instances += d['instance']
        img = img.cuda()
        vdo = vdo.cuda()
        with torch.no_grad():
            img_f = model(img).cpu().numpy()
            vdo_f = model(vdo).cpu().numpy()

        img_features = np.append(img_features, img_f, axis=0)
        vdo_features = np.append(vdo_features, vdo_f, axis=0)

    with open('data/instance2label.json', 'r') as f:
        ins2labDic = json.load(f)

    # rates, acc = cal_cosine_similarity(vdo_features, img_features, instances, ins2labDic)
    rates, acc = kmeans_classifer(opt, vdo_features, img_features, instances, ins2labDic)
    print(sum(rates)/len(rates), min(rates), max(rates))
    print(acc)

if __name__ == "__main__":
    import torchvision.transforms as transforms
    opt = get_args_arcface()
    evaluate(opt)
    # kmeans = joblib.load(os.path.join(opt.saved_path, 'kmeans.m'))
    # l = os.listdir('data/train_instance')
    # ll = []
    # for n in tqdm(l):
    #     if '317900' in n:
    #         ll.append(n)
    # imgs = []
    # for n in ll:
    #     img = np.load(os.path.join('data/train_instance/', n)[:-4]+'.npy')

    #     img = torch.from_numpy(img)
    #     img = img.permute(2, 0, 1)
    #     transform = transforms.Normalize(
    #         mean=[0.55574415, 0.51230767, 0.51123354], 
    #         std=[0.21303795, 0.21604613, 0.21273348])
    #     img = transform(img)
    #     imgs.append(img.unsqueeze(0))
    # img = torch.cat(imgs)
    # model = Backbone(opt)
    
    # b_name = 'backbone_'+opt.mode+'_{}'.format(opt.num_layers)

    # model.load_state_dict(torch.load(os.path.join(opt.saved_path, b_name+'.pth')))
    # model.cuda()
    # model.eval()
    # img = img.cuda()
    # with torch.no_grad():
    #     features = model(img).cpu().numpy()
    # print(ll)
    # print(kmeans.predict(features))



