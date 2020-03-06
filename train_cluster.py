import os
import torch
from torch.utils.data import DataLoader
from sklearn.cluster import MiniBatchKMeans, KMeans
from arcface.backbone import Backbone
from dataset import ArcfaceDataset
from config import get_args_arcface
import joblib
import numpy as np
from tqdm import tqdm

def train_KMeans(opt):
    model = Backbone(opt)
    
    b_name = 'backbone_'+opt.mode+'_{}'.format(opt.num_layers)

    opt.batch_size *= 8
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
    d = {}
    print('predicting train data...')
    for data in tqdm(training_generator):
        img = data['img'].cuda()
        with torch.no_grad():
            embedding = model(img).cpu().numpy()
        for i, label in enumerate(data['label'].numpy()):
            if label not in d:
                d[label] = embedding[i].reshape(1, -1)
            else:
                d[label] = np.append(d[label], embedding[i].reshape(1, -1), axis=0)
        features = np.append(features, embedding, axis=0)
    torch.cuda.empty_cache()

    print('calculating center points...')
    centerPoint = np.zeros((0, opt.embedding_size))
    for v in tqdm(d.values()):
        p = np.mean(v, axis=0)
        centerPoint = np.append(centerPoint, p.reshape(1, -1), axis=0)
    
    print('creating kmeans...')
    kmeans = MiniBatchKMeans(n_clusters=num_classes, batch_size=20000, verbose=1, init=centerPoint, tol=1e-8)
    # kmeans = KMeans(n_clusters=num_classes, n_jobs=8, verbose=1, init=centerPoint, tol=1e-8)
    kmeans.fit(features)

    joblib.dump(kmeans, os.path.join(opt.saved_path, 'kmeans.m'))
    

if __name__ == "__main__":
    opt = get_args_arcface()
    train_KMeans(opt)