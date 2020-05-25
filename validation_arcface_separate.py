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
from arcface.resnest import ResNeSt, PreModule
from arcface.efficientnet import EfficientNet
from config import get_args_arcface
from dataset import ValidationArcfaceDataset
from tqdm import tqdm
# from sklearn.metrics.pairwise import cosine_similarity
# from joint_bayesian.JointBayesian import verify
# import joblib
import json

def cosine_similarity(a, b):
    return a@b.T

# def kmeans_classifer(opt, vdo_features, img_features, instances, ins2labDic):
#     print('Classifying with k-means...')
#     kmeans = joblib.load(os.path.join(opt.saved_path, 'kmeans.m'))
#     vdo_cls = kmeans.predict(vdo_features)
#     img_cls = kmeans.predict(img_features)

#     vdo_cls = np.array(vdo_cls)
#     img_cls = np.array(img_cls)

#     acc = 0
#     miss = 0
#     rates_t = []
#     rates_f = []
#     for i, vc in enumerate(tqdm(vdo_cls)):
#         l = []
#         arg = np.argwhere(img_cls==vc).reshape(-1)
#         for j in arg:
#             if ins2labDic[instances[i]] == ins2labDic[instances[j]]:
#                 similarity = vdo_features[i]@img_features[j].T
#                 l.append([j, similarity])
#         if len(l) == 0:
#             miss += 1
#             continue
#         m = max(l, key=lambda x: x[1])
#         if m[0] == i:
#             acc += 1
#             rates_t.append(m[1])
#         else:
#             rates_f.append(m[1])
#     print(miss/len(vdo_cls))
#     return rates_t, rates_f, acc/len(vdo_cls)


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

# def joint_bayesian(opt, vdo_features, img_features, instances, ins2labDic):
#     print('Calculating Joint Bayesian...')
#     G = np.load(os.path.join(opt.saved_path, 'G.npy'))
#     A = np.load(os.path.join(opt.saved_path, 'A.npy'))

#     scores = verify(A, G, vdo_features, img_features)

#     argmax = np.argsort(-scores, axis=1)
#     acc = 0
#     rates_t = []
#     rates_f = []
#     for i in tqdm(range(len(scores))):
#         for j in argmax[i]:
#             if ins2labDic[instances[i]] != ins2labDic[instances[j]]:
#                 continue
#             if j == i:
#                 acc +=1
#                 rates_t.append(scores[i, j])
#             else:
#                 rates_f.append(scores[i, j])
#             break
#     return rates_t, rates_f, acc/len(scores)


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
        model_image = ResNetCBAM(opt)
        model_video = ResNetCBAM(opt)
        b_name = opt.network+'_{}'.format(opt.num_layers_c)
        b_name_image = b_name + '_image'
        b_name_video = b_name + '_video'
    elif opt.network == 'resnest':
        pre_model_image = PreModule(opt)
        pre_model_video = PreModule(opt)
        model = ResNeSt(opt)
        b_name = opt.network+'_{}'.format(opt.num_layers_s)
        p_name_image = 'pre_'+b_name+'_image'
        p_name_video = 'pre_'+b_name+'_video'
    elif 'efficientnet' in opt.network:
        model = EfficientNet(opt)
        b_name = opt.network
        h_name = 'arcface_'+b_name
    else:
        raise RuntimeError('Cannot Find the Model: {}'.format(opt.network))

    print(b_name)

    pre_model_image.load_state_dict(torch.load(os.path.join(opt.saved_path, p_name_image+'.pth')))
    pre_model_image.cuda()
    pre_model_image.eval()

    pre_model_video.load_state_dict(torch.load(os.path.join(opt.saved_path, p_name_video+'.pth')))
    pre_model_video.cuda()
    pre_model_video.eval()

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
            img = pre_model_image(img)
            vdo = pre_model_video(vdo)
            img_f = model(img)
            vdo_f = model(vdo)

        img_f = img_f.cpu().numpy()
        vdo_f = vdo_f.cpu().numpy()

        img_features = np.append(img_features, img_f, axis=0)
        vdo_features = np.append(vdo_features, vdo_f, axis=0)

    print('Calculating cosine similarity...')
    cos = cosine_similarity(vdo_features, img_features)
    return cos, instances

if __name__ == "__main__":
    opt = get_args_arcface()
    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(opt.GPUs[0])
    opt.batch_size *= 2
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



