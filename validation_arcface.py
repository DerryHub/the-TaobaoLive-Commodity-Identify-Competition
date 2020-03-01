import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from arcface.backbone import Backbone
from arcface.head import Arcface
from config import get_args_arcface
from dataset import ValidationArcfaceDataset
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import json


def evaluate(opt):
    dataset = ValidationArcfaceDataset(root_dir='data/validation_instance/')
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=8)

    model = Backbone(opt)
    
    b_name = 'backbone_'+opt.mode+'_{}'.format(opt.num_layers)

    model.load_state_dict(torch.load(os.path.join(opt.saved_path, b_name+'.pth')))
    model.cuda()
    model.eval()

    img_features = np.zeros((0, 512))
    vdo_features = np.zeros((0, 512))
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

    print('Calculating cosine similarity...')
    cos = cosine_similarity(vdo_features, img_features)
    acc = 0
    rates = []
    for i in tqdm(range(len(cos))):
        l = [[cos[i, j], instances[j], j] for j in range(len(cos))]
        l = sorted(l, key=lambda x: x[0], reverse=True)
        for item in l:
            if ins2labDic[instances[i]] != ins2labDic[item[1]]:
                continue
            if item[2] == i:
                acc += 1
            rates.append(item[0])
            break
    print(sum(rates)/len(rates), min(rates), max(rates))
    print(acc/len(cos))

if __name__ == "__main__":
    opt = get_args_arcface()
    evaluate(opt)