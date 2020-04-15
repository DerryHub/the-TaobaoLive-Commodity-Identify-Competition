import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from arcface.resnet import ResNet
from arcface.googlenet import GoogLeNet
from arcface.inception_v4 import InceptionV4
from arcface.inceptionresnet_v2 import InceptionResNetV2
from arcface.densenet import DenseNet
from arcface.resnet_cbam import ResNetCBAM
from config import get_args_arcface
from dataset import ValidationArcfaceDataset, ArcfaceDataset
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from joint_bayesian.JointBayesian import verify
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
    rates_t = []
    rates_f = []
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
            rates_t.append(m[1])
        else:
            rates_f.append(m[1])
    print(miss/len(vdo_cls))
    return rates_t, rates_f, acc/len(vdo_cls)


def cal_cosine_similarity(vdo_features, img_features, instances, ins2labDic):
    print('Calculating cosine similarity...')
    cos = cosine_similarity(vdo_features, img_features)
    # argmax = np.argsort(-cos, axis=1)
    acc = 0
    rates_t = []
    rates_f = []
    length = len(instances) // 2
    cos = np.max((cos[:length, :length], cos[length:, length:], cos[length:, :length], cos[:length, length:]), axis=0)
    argmax = np.argsort(-cos, axis=1)
    for i in tqdm(range(len(cos))):
        for j in argmax[i]:
            if ins2labDic[instances[i]] != ins2labDic[instances[j]]:
                continue
            # if j%length == i%length:
            if i==j:
                acc +=1
                rates_t.append(cos[i, j])
            else:
                rates_f.append(cos[i, j])
            break
    return rates_t, rates_f, acc/len(cos)

def joint_bayesian(opt, vdo_features, img_features, instances, ins2labDic):
    print('Calculating Joint Bayesian...')
    G = np.load(os.path.join(opt.saved_path, 'G.npy'))
    A = np.load(os.path.join(opt.saved_path, 'A.npy'))

    scores = verify(A, G, vdo_features, img_features)

    argmax = np.argsort(-scores, axis=1)
    acc = 0
    rates_t = []
    rates_f = []
    for i in tqdm(range(len(scores))):
        for j in argmax[i]:
            if ins2labDic[instances[i]] != ins2labDic[instances[j]]:
                continue
            if j == i:
                acc +=1
                rates_t.append(scores[i, j])
            else:
                rates_f.append(scores[i, j])
            break
    return rates_t, rates_f, acc/len(scores)


def evaluate(opt):
    dataset = ValidationArcfaceDataset(root_dir='data/validation_instance/', size=(opt.size, opt.size))
    loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=4)

    if opt.network == 'resnet':
        model = ResNet(opt)
        b_name = opt.network+'_'+opt.mode+'_{}'.format(opt.num_layers_r)
    elif opt.network == 'googlenet':
        model = GoogLeNet(opt)
        b_name = opt.network
    elif opt.network == 'inceptionv4':
        model = InceptionV4(opt)
        b_name = opt.network
    elif opt.network == 'inceptionresnetv2':
        model = InceptionResNetV2(opt)
        b_name = opt.network
    elif opt.network == 'densenet':
        model = DenseNet(opt)
        b_name = opt.network+'_{}'.format(opt.num_layers_d)
    elif opt.network == 'resnet_cbam':
        model = ResNetCBAM(opt)
        b_name = opt.network+'_{}'.format(opt.num_layers_c)
    else:
        raise RuntimeError('Cannot Find the Model: {}'.format(opt.network))

    model.load_state_dict(torch.load(os.path.join(opt.saved_path, b_name+'.pth')))
    model.cuda()
    model.eval()

    img_features = np.zeros((0, opt.embedding_size))
    vdo_features = np.zeros((0, opt.embedding_size))
    instances = []

    print('Predecting...')
    for d in tqdm(loader):
        img = d['img'].cuda()
        vdo = d['vdo'].cuda()
        instances += d['instance']
        img_text = d['img_text'].cuda()
        vdo_text = d['vdo_text'].cuda()
        img_e = d['img_e'].cuda()
        vdo_e = d['vdo_e'].cuda()
        with torch.no_grad():
            img_f = model(img)
            vdo_f = model(vdo)

        img_f = img_f.cpu().numpy()
        vdo_f = vdo_f.cpu().numpy()

        img_features = np.append(img_features, img_f, axis=0)
        vdo_features = np.append(vdo_features, vdo_f, axis=0)

    # with open('data/instance2label.json', 'r') as f:
    #     ins2labDic = json.load(f)

    print('Calculating cosine similarity...')
    cos = cosine_similarity(vdo_features, img_features)
    return cos, instances

    # rates_t, rates_f, acc = cal_cosine_similarity(vdo_features, img_features, instances, ins2labDic)
    # # rates_t, rates_f, acc = joint_bayesian(opt, vdo_features, img_features, instances, ins2labDic)
    # # rates, acc = kmeans_classifer(opt, vdo_features, img_features, instances, ins2labDic)
    # print(sum(rates_t)/len(rates_t), min(rates_t), max(rates_t))
    # print(sum(rates_f)/len(rates_f), min(rates_f), max(rates_f))
    # print(acc)

if __name__ == "__main__":
    opt = get_args_arcface()
    opt.batch_size *= 4
    config_list = opt.validation_config
    cos = 0
    for network, size, num_layers, r in config_list:
        opt.network = network
        opt.size = size
        opt.num_layers_c = num_layers
        opt.num_layers_r = num_layers
        cos_, instances = evaluate(opt)
        cos += cos_ * r
    with open('data/instance2label.json', 'r') as f:
        ins2labDic = json.load(f)
    acc = 0
    rates_t = []
    rates_f = []
    length = len(instances) // 2
    # cos_1 = np.min((cos[:length, :length], cos[length:, length:], cos[length:, :length], cos[:length, length:]), axis=0)
    cos = np.max((cos[:length, :length], cos[length:, length:], cos[length:, :length], cos[:length, length:]), axis=0)
    # cos = cos_1 + cos_2
    argmax = np.argsort(-cos, axis=1)
    for i in tqdm(range(len(cos))):
        for j in argmax[i]:
            if ins2labDic[instances[i]] != ins2labDic[instances[j]]:
                continue
            # if j%length == i%length:
            if i==j:
                acc +=1
                rates_t.append(cos[i, j])
            else:
                rates_f.append(cos[i, j])
            break

    print(sum(rates_t)/len(rates_t), min(rates_t), max(rates_t))
    print(sum(rates_f)/len(rates_f), min(rates_f), max(rates_f))
    print(acc/len(cos))
    # # kmeans = joblib.load(os.path.join(opt.saved_path, 'kmeans.m'))
    # training_set = ArcfaceDataset(root_dir=opt.data_path, mode="train", size=(opt.size, opt.size))
    # opt.num_classes = training_set.num_classes
    # l = os.listdir('data/train_instance')
    # ll = []
    # # print(l[:100])
    # for n in tqdm(l):
    #     if '219636' in n:
    #         ll.append(n)
    # imgs = []
    # for n in ll:
    #     img = np.load(os.path.join('data/train_instance/', n)[:-4]+'.npy')
    #     img = img[7:112+7, 7:112+7, :]
    #     img = torch.from_numpy(img)
    #     img = img.permute(2, 0, 1)
    #     transform = transforms.Normalize(
    #         mean=[0.55574415, 0.51230767, 0.51123354], 
    #         std=[0.21303795, 0.21604613, 0.21273348])
    #     img = transform(img)
    #     imgs.append(img.unsqueeze(0))
    # img = torch.cat(imgs)
    # # features = []
    # # for i in img:
    # #     features.append(SIFT(i))
    # # dis = []
    # # for i in range(10):
    # #     dis.append(hamming_distance(features[0], features[i]))
    # # print(dis)
    # # print(img.size())
    # model = InceptionResNetV2(opt)
    
    # b_name = opt.network
    # model.load_state_dict(torch.load(os.path.join(opt.saved_path, b_name+'.pth')))
    # model.cuda()
    # model.eval()
    # img = img.cuda()
    # with torch.no_grad():
    #     features = model(img)
    # from arcface.head import Arcface
    # opt.m = 0
    # h = Arcface(opt)
    # h_name = 'arcface_'+b_name
    # h.load_state_dict(torch.load(os.path.join(opt.saved_path, h_name+'.pth')))
    # h.cuda()
    # h.eval()
    # o = h(features).cpu()
    # o = torch.argmax(o, dim=1)
    # print(o)
    # # # # print(ll)
    # # print(kmeans.predict(features))



