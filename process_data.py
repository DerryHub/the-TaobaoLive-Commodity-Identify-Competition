import os
import json
from shutil import copyfile, rmtree
from tqdm import tqdm
import cv2
import numpy as np

def processImage(img_tat, root, annotation, label):
    img_ann_path = os.path.join(root, 'image_annotation/')
    img_path = os.path.join(root, 'image/')
    dirList = os.listdir(img_ann_path)

    dirList_t = tqdm(dirList)
    dirList_t.set_description_str('    processing image data')
    for filename in dirList_t:
        jsonList = os.listdir(os.path.join(img_ann_path, filename))
        for jsonname in jsonList:
            with open(os.path.join(img_ann_path, filename, jsonname), 'r') as f:
                dic = json.load(f)
            new_name = dic['item_id']+'_'+dic['img_name']
            copyfile(os.path.join(img_path, dic['item_id'], dic['img_name']), os.path.join(img_tat, new_name))
            d = {}
            d['img_name'] = new_name
            d['annotations'] = dic['annotations']
            for i, ann in enumerate(d['annotations']):
                if ann['label'] == '古装':
                    ann['label'] = '古风'
                if ann['label'] not in label['label2index'].keys():
                    label['label2index'][ann['label']] = len(label['label2index'])
                    label['index2label'][len(label['index2label'])] = ann['label']
                d['annotations'][i]['label'] = label['label2index'][ann['label']]
            annotation['annotations'].append(d)
    return annotation, label

def processVideo(vdo_tat, root, annotation, label):
    vdo_ann_path = os.path.join(root, 'video_annotation/')
    vdo_path = os.path.join(root, 'video/')
    jsonList = os.listdir(vdo_ann_path)

    jsonList_t = tqdm(jsonList)
    jsonList_t.set_description_str('    processing video data')
    for filename in jsonList_t:
        with open(os.path.join(vdo_ann_path, filename), 'r') as f:
            dic = json.load(f)
        frame_indexs = [d['frame_index'] for d in dic['frames']]
        annotation_dic = {}
        for d in dic['frames']:
            for i, ann in enumerate(d['annotations']):
                if ann['label'] == '古装':
                    ann['label'] = '古风'
                if ann['label'] not in label['label2index'].keys():
                    label['label2index'][ann['label']] = len(label['label2index'])
                    label['index2label'][len(label['index2label'])] = ann['label']
                d['annotations'][i]['label'] = label['label2index'][ann['label']]
            annotation_dic[d['frame_index']] = d['annotations']
        vdo_name = dic['video_id']+'.mp4'
        cap = cv2.VideoCapture(os.path.join(vdo_path, vdo_name))
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        for i in range(int(frames)):
            ret, frame = cap.read()
            if i in frame_indexs:
                d = {}
                img_name = dic['video_id']+'_{}.jpg'.format(i)
                d['img_name'] = img_name
                d['annotations'] = annotation_dic[i]
                cv2.imwrite(os.path.join(vdo_tat, img_name), frame)
                annotation['annotations'].append(d)
    return annotation, label

def processTrain(label):
    roots = [1, 2, 3, 4, 5, 6]
    img_tat = 'data/train_images'
    vdo_tat = 'data/train_videos'

    label_file = 'data/label.json'

    if os.path.isdir(img_tat):
        rmtree(img_tat)
    os.makedirs(img_tat)

    if os.path.isdir(vdo_tat):
        rmtree(vdo_tat)
    os.makedirs(vdo_tat)

    annotation_image = {}
    annotation_image['annotations'] = []

    annotation_video = {}
    annotation_video['annotations'] = []
    
    for i in roots:
        root = 'data/train_dataset_part{}'.format(i)
        print('processing train data [{}/{}]:'.format(i, len(roots)))
        annotation_image, label = processImage(img_tat, root, annotation_image, label)
        annotation_video, label = processVideo(vdo_tat, root, annotation_video, label)

    with open(img_tat+'_annotation.json', 'w') as f:
        json.dump(annotation_image, f)
    
    with open(vdo_tat+'_annotation.json', 'w') as f:
        json.dump(annotation_video, f)
    
    with open(label_file, 'w') as f:
        json.dump(label, f)

def processValidation(label):
    roots = [1, 2]
    img_tat = 'data/validation_images'
    vdo_tat = 'data/validation_videos' 

    if os.path.isdir(img_tat):
        rmtree(img_tat)
    os.makedirs(img_tat)

    if os.path.isdir(vdo_tat):
        rmtree(vdo_tat)
    os.makedirs(vdo_tat)

    annotation_image = {}
    annotation_image['annotations'] = []

    annotation_video = {}
    annotation_video['annotations'] = []
    
    for i in roots:
        root = 'data/validation_dataset_part{}'.format(i)
        print('processing validation data [{}/{}]:'.format(i, len(roots)))
        annotation_image, label = processImage(img_tat, root, annotation_image, label)
        annotation_video, label = processVideo(vdo_tat, root, annotation_video, label)
    
    with open(img_tat+'_annotation.json', 'w') as f:
        json.dump(annotation_image, f)
    
    with open(vdo_tat+'_annotation.json', 'w') as f:
        json.dump(annotation_video, f)


def saveNumpyInstance(root_dir, mode, size):
    img_tat = mode + '_images'
    vdo_tat = mode + '_videos'
    savePath = mode + '_instance'
    savePath = os.path.join(root_dir, savePath)

    # if os.path.isdir(savePath):
    #     rmtree(savePath)
    # os.makedirs(savePath)

    with open(os.path.join(root_dir, img_tat+'_annotation.json'), 'r') as f:
        d_i = json.load(f)
    with open(os.path.join(root_dir, vdo_tat+'_annotation.json'), 'r') as f:
        d_v = json.load(f)

    l_i = d_i['annotations']
    l_v = d_v['annotations']

    images = []

    for d in l_i:
        for dd in d['annotations']:
            if dd['instance_id'] > 0:
                pass
                # t = []
                # t.append(os.path.join(img_tat, d['img_name']))
                # t.append(img_tat+str(dd['instance_id'])+d['img_name'])
                # t.append(dd['box'])
                # t.append(dd['instance_id'])
                # images.append(t)
            else:
                t = []
                t.append(os.path.join(img_tat, d['img_name']))
                t.append(img_tat+str(dd['instance_id'])+str(dd['label'])+d['img_name'])
                t.append(dd['box'])
                t.append(dd['instance_id'])
                images.append(t)

    for d in l_v:
        for dd in d['annotations']:
            if dd['instance_id'] > 0:
                pass
                # t = []
                # t.append(os.path.join(vdo_tat, d['img_name']))
                # t.append(vdo_tat+str(dd['instance_id'])+d['img_name'])
                # t.append(dd['box'])
                # t.append(dd['instance_id'])
                # images.append(t)
            else:
                t = []
                t.append(os.path.join(vdo_tat, d['img_name']))
                t.append(vdo_tat+str(dd['instance_id'])+str(dd['label'])+d['img_name'])
                t.append(dd['box'])
                t.append(dd['instance_id'])
                images.append(t)
    
    if mode == 'train':
        for imgPath, saveName, box, instance_id in tqdm(images):
            img = cv2.imread(os.path.join(root_dir, imgPath))
            h, w, c = img.shape
            dh = int((box[3]-box[1])*0.1)
            dw = int((box[2]-box[0])*0.1)
            box[1] = max(0, box[1]-dh)
            box[3] = min(h, box[3]+dh)
            box[0] = max(0, box[0]-dw)
            box[2] = min(w, box[2]+dw)
            img = img[box[1]:box[3], box[0]:box[2], :]
            img = cv2.resize(img, size)
            # h, w, c = img.shape
            # if h > w:
            #     nh = size
            #     nw = size*w//h
            # else:
            #     nw = size
            #     nh = size*h//w
            # img = cv2.resize(img, (nw, nh))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255
            np.save(os.path.join(savePath, saveName)[:-4]+'.npy', img)
    elif mode == 'validation':
        for imgPath, saveName, box, instance_id in tqdm(images):
            if not os.path.exists(os.path.join(savePath, str(instance_id))):
                os.mkdir(os.path.join(savePath, str(instance_id)))
            img = cv2.imread(os.path.join(root_dir, imgPath))
            img = img[box[1]:box[3], box[0]:box[2], :]
            img = cv2.resize(img, size)
            # h, w, c = img.shape
            # if h > w:
            #     nh = size
            #     nw = size*w//h
            # else:
            #     nw = size
            #     nh = size*h//w
            # img = cv2.resize(img, (nw, nh))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255
            np.save(os.path.join(savePath, str(instance_id), saveName)[:-4]+'.npy', img)


def createInstance2Label(root_dir):
    items = []

    modes = ['train', 'validation']
    for mode in modes:
        img_tat = mode + '_images'
        vdo_tat = mode + '_videos'

        with open(os.path.join(root_dir, img_tat+'_annotation.json'), 'r') as f:
            d_i = json.load(f)
        with open(os.path.join(root_dir, vdo_tat+'_annotation.json'), 'r') as f:
            d_v = json.load(f)

        l_i = d_i['annotations']
        l_v = d_v['annotations']

        for d in l_i:
            for dd in d['annotations']:
                if dd['instance_id'] > 0:
                    t = []
                    t.append(dd['label'])
                    t.append(dd['instance_id'])
                    items.append(t)

        for d in l_v:
            for dd in d['annotations']:
                if dd['instance_id'] > 0:
                    t = []
                    t.append(dd['label'])
                    t.append(dd['instance_id'])
                    items.append(t)

    ins2labDic = {}
    for label, instance_id in tqdm(items):
        if instance_id in ins2labDic:
            continue
        ins2labDic[instance_id] = label

    with open(os.path.join(root_dir, 'instance2label.json'), 'w') as f:
        json.dump(ins2labDic, f)



if __name__ == "__main__":
#     label = {}
#     label['label2index'] = {}
#     label['index2label'] = {}
#     processTrain(label)
#     processValidation(label)
    saveNumpyInstance('data', 'train', (128, 128))
    saveNumpyInstance('data', 'validation', (112, 112))
#     saveNumpyInstance('data', 'validation', (112, 112))
#     saveNumpyImage('data', 'train')
#     saveNumpyImage('data', 'validation')
#     createInstance2Label('data')
    

