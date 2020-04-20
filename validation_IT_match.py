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
from text.TF_IDF import TF_IDF
from text.BERT import BERT
from config import get_args_arcface
from dataset import ITMatchValidation
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import json


opt = get_args_arcface()

dataset = ITMatchValidation(root_dir='data/validation_instance/', size=(224, 224))
loader = DataLoader(dataset, batch_size=30, shuffle=False, num_workers=12)

opt.vocab_size = dataset.vocab_size

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

if opt.network_text == 'bert':
    model_text = BERT(opt)
    model_name = 'BERT_'+b_name
elif opt.network_text == 'tf_idf':
    model_text = TF_IDF(opt)
    model_name = 'TFIDF_'+b_name

print(model_name)

model_text.load_state_dict(torch.load(os.path.join(opt.saved_path, model_name+'.pth')))
model_text.cuda()
model_text.eval()

img_features = np.zeros((0, opt.embedding_size))
text_features = np.zeros((0, opt.embedding_size))

print('Predecting...')
acc = 0
total = 0
for d in tqdm(loader):
    img = d['img'].cuda()
    text = d['text'].cuda()
    with torch.no_grad():
        img_f = model(img).cpu().numpy()
        text_f = model_text(text).cpu().numpy()

    cos = cosine_similarity(text_f, img_f)
    argmax = np.argsort(-cos, axis=1)
    for i in range(len(cos)):
        if argmax[i][0] == i:
            acc += 1
    total += len(cos)
    # img_features = np.append(img_features, img_f, axis=0)
    # text_features = np.append(text_features, text_f, axis=0)
    print(acc/total)
# print('Calculating cosine similarity...')
# cos = cosine_similarity(text_features, img_features)

# acc = 0
# rates_t = []
# rates_f = []
# argmax = np.argsort(-cos, axis=1)
# for i in tqdm(range(len(cos))):
#     if argmax[i][0] == i:
#         acc +=1
#         rates_t.append(cos[i, argmax[i][0]])
#     else:
#         rates_f.append(cos[i, argmax[i][0]])

# print(sum(rates_t)/len(rates_t), min(rates_t), max(rates_t))
# print(sum(rates_f)/len(rates_f), min(rates_f), max(rates_f))
# print(acc/len(cos))