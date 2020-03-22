import os
import torch
from torch.utils.data import DataLoader
from arcface.resnet import ResNet
from joint_bayesian.JointBayesian import JointBayesian_Train
from dataset import ArcfaceDataset
from config import get_args_arcface
import numpy as np
from tqdm import tqdm

def train(opt):
    print(opt)
    model = ResNet(opt)
        
    b_name = opt.network+'_'+opt.mode+'_{}'.format(opt.num_layers)

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

    features = np.zeros((0, opt.embedding_size))
    labels = np.zeros((0))
    print('predicting train data...')
    for data in tqdm(training_generator):
        img = data['img'].cuda()
        with torch.no_grad():
            embedding = model(img).cpu().numpy()
        labels = np.append(labels, data['label'].numpy())
        features = np.append(features, embedding, axis=0)
    torch.cuda.empty_cache()

    JointBayesian_Train(features, labels, opt.saved_path)


if __name__ == "__main__":
    opt = get_args_arcface()
    train(opt)