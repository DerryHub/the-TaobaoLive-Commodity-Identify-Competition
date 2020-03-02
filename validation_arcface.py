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
from sklearn.cluster import MiniBatchKMeans
import joblib
import json

def save_KMeans(opt):
    model = Backbone(opt)
    
    b_name = 'backbone_'+opt.mode+'_{}'.format(opt.num_layers)

    opt.batch_size *= 4
    model.load_state_dict(torch.load(os.path.join(opt.saved_path, b_name+'.pth')))
    model.cuda()
    model.eval()

    training_params = {"batch_size": opt.batch_size,
                        "shuffle": True,
                        "drop_last": False,
                        "num_workers": opt.workers}

    training_set = ArcfaceDataset(root_dir=opt.data_path, mode="train", size=(opt.size, opt.size))
    training_generator = DataLoader(training_set, **training_params)
    num_classes = training_set.num_classes

    features = np.zeros((0, opt.embedding_size))

    print('predicting train data...')
    for data in tqdm(training_generator):
        img = data['img'].cuda()
        with torch.no_grad():
            embedding = model(img).cpu().numpy()
        features = np.append(features, embedding, axis=0)
    torch.cuda.empty_cache()

    print('creating kmeans...')
    kmeans = MiniBatchKMeans(n_clusters=num_classes, batch_size=20000, verbose=1)
    kmeans.fit(features)

    joblib.dump(kmeans, os.path.join(opt.saved_path, 'kmeans.m'))
    

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

    rates, acc = cal_cosine_similarity(vdo_features, img_features, instances, ins2labDic)
    print(sum(rates)/len(rates), min(rates), max(rates))
    print(acc)

if __name__ == "__main__":
    opt = get_args_arcface()
    save_KMeans(opt)
    # evaluate(opt)