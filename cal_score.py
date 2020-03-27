import os
import json
import torch
import numpy as np
from config import get_args_arcface, get_args_efficientdet
from utils import iou
from tqdm import tqdm

opt_a = get_args_arcface()
opt_e = get_args_efficientdet()

with open('result.json', 'r') as f:
    dic = json.load(f)

with open(os.path.join(opt_a.data_path, 'validation_images_annotation.json'), 'r') as f:
    img_lst = json.load(f)['annotations']
with open(os.path.join(opt_a.data_path, 'validation_videos_annotation.json'), 'r') as f:
    vdo_lst = json.load(f)['annotations']

img_dic = {}
vdo_dic = {}
for d in img_lst:
    img_dic[d['img_name'][:-4]] = d['annotations']
for d in vdo_lst:
    vdo_dic[d['img_name'][:-4]] = d['annotations']

N_TP = [0]*3
N_P = [len(dic)]*3
N_GT = [10000]*3

for k in tqdm(dic.keys()):
    if k == dic[k]['item_id']:
        N_TP[0] += 1
        flag_1 = False
        flag_2 = False
        for result in dic[k]['result']:
            img_box = []
            instances = []
            for annotation in img_dic[dic[k]['item_id']+ '_'+result['img_name']]:
                instance = annotation['instance_id']
                if instance > 0:
                    instances.append(instance)
                    img_box.append(torch.from_numpy(np.array(annotation['box'])))

            flag_1 = True
            flag_2 = True
            f = True
            for annotation in vdo_dic[k+ '_'+str(dic[k]['frame_index'])]:
                instance = annotation['instance_id']
                if instance not in instances:
                    flag_1 = False
                    flag_2 = False
                    f = False
                else:
                    for box in img_box:
                        IOU = iou(torch.tensor(result['item_box']).unsqueeze(0), box.unsqueeze(0))
                        if IOU > 0.5 and f:
                            flag_2 = True
                            break
                        flag_2 = False
                    if flag_2 == False:
                        f = False
                    
        if flag_1:
            N_TP[1] += 1
        if flag_2:
            N_TP[2] += 1


p = [N_TP[i]/N_P[i] for i in range(3)]
r = [N_TP[i]/N_GT[i] for i in range(3)]
s = [(2*p[i]*r[i])/(p[i]+r[i]) for i in range(3)]

for i in range(3):
    print('#######################################')
    print('N_P {}: {}\tN_GT {}: {}\tN_TP {}: {}'.format(i+1, N_P[i], i+1, N_GT[i], i+1, N_TP[i]))
    print('p {}: {}\tr {}: {}'.format(i+1, p[i], i+1, r[i]))
    print('s {}: {}'.format(i+1, s[i]))
    print()