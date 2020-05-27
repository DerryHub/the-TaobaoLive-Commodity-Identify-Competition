import os
import json
from shutil import copyfile, rmtree
from tqdm import tqdm
import cv2
import numpy as np
import jieba

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
        for i in frame_indexs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            d = {}
            img_name = dic['video_id']+'_{}.jpg'.format(i)
            d['img_name'] = img_name
            d['annotations'] = annotation_dic[i]
            cv2.imwrite(os.path.join(vdo_tat, img_name), frame)
            annotation['annotations'].append(d)
        # for i in range(int(frames)):
        #     ret, frame = cap.read()
        #     if i in frame_indexs:
        #         d = {}
        #         img_name = dic['video_id']+'_{}.jpg'.format(i)
        #         d['img_name'] = img_name
        #         d['annotations'] = annotation_dic[i]
        #         cv2.imwrite(os.path.join(vdo_tat, img_name), frame)
        #         annotation['annotations'].append(d)
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

def processValidation_2(label):
    roots = [3, 4]
    img_tat = 'data/validation_2_images'
    vdo_tat = 'data/validation_2_videos' 

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

    if os.path.isdir(savePath):
        rmtree(savePath)
    os.makedirs(savePath)

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
                t = []
                t.append(os.path.join(img_tat, d['img_name']))
                t.append(img_tat+str(dd['instance_id'])+d['img_name'])
                t.append(dd['box'])
                t.append(dd['instance_id'])
                images.append(t)

    for d in l_v:
        for dd in d['annotations']:
            if dd['instance_id'] > 0:
                t = []
                t.append(os.path.join(vdo_tat, d['img_name']))
                t.append(vdo_tat+str(dd['instance_id'])+d['img_name'])
                t.append(dd['box'])
                t.append(dd['instance_id'])
                images.append(t)
    
    for imgPath, saveName, box, instance_id in tqdm(images):
        if not os.path.exists(os.path.join(savePath, str(instance_id))):
            os.mkdir(os.path.join(savePath, str(instance_id)))
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
        # cv2.imwrite(os.path.join(savePath, str(instance_id), saveName)[:-4]+'.jpg', img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255
        np.save(os.path.join(savePath, str(instance_id), saveName)[:-4]+'.npy', img)


def createInstance2Label(root_dir):
    items = []

    modes = ['train', 'validation', 'validation_2']
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

def createInstanceID(root_dir='data'):
    mode = 'train'
    img_tat = mode + '_images'
    vdo_tat = mode + '_videos'

    with open(os.path.join(root_dir, img_tat+'_annotation.json'), 'r') as f:
        d_i = json.load(f)
    with open(os.path.join(root_dir, vdo_tat+'_annotation.json'), 'r') as f:
        d_v = json.load(f)

    l_i = d_i['annotations']
    l_v = d_v['annotations']

    instance = {}
    s_i = set([])
    s_v = set([])

    clsDic = {}

    for d in l_i:
        for dd in d['annotations']:
            if dd['instance_id'] > 0:
                s_i.add(dd['instance_id'])
                if dd['instance_id'] not in instance:
                    instance[dd['instance_id']] = 1
                else:
                    instance[dd['instance_id']] += 1

    for d in l_v:
        for dd in d['annotations']:
            if dd['instance_id'] > 0:
                s_v.add(dd['instance_id'])
                if dd['instance_id'] not in instance:
                    instance[dd['instance_id']] = 1
                else:
                    instance[dd['instance_id']] += 1

    id_set = s_i & s_v
    all_ids = set([])
    
    for ID in id_set:
        if instance[ID] > 10 and instance[ID] < 20:
            all_ids.add(ID)

    for i in all_ids:
        clsDic[i] = len(clsDic)

    with open(os.path.join(root_dir, 'instanceID.json'), 'w') as f:
        json.dump(clsDic, f)

def createInstanceID_2(root_dir='data'):
    modes = ['train', 'validation_2']

    all_ids = set([])
    for mode in modes:
        img_tat = mode + '_images'
        vdo_tat = mode + '_videos'

        with open(os.path.join(root_dir, img_tat+'_annotation.json'), 'r') as f:
            d_i = json.load(f)
        with open(os.path.join(root_dir, vdo_tat+'_annotation.json'), 'r') as f:
            d_v = json.load(f)

        l_i = d_i['annotations']
        l_v = d_v['annotations']

        instance = {}
        s_i = set([])
        s_v = set([])

        for d in l_i:
            for dd in d['annotations']:
                if dd['instance_id'] > 0:
                    s_i.add(dd['instance_id'])
                    if dd['instance_id'] not in instance:
                        instance[dd['instance_id']] = 1
                    else:
                        instance[dd['instance_id']] += 1

        for d in l_v:
            for dd in d['annotations']:
                if dd['instance_id'] > 0:
                    s_v.add(dd['instance_id'])
                    if dd['instance_id'] not in instance:
                        instance[dd['instance_id']] = 1
                    else:
                        instance[dd['instance_id']] += 1

        id_set = s_i & s_v
        
        for ID in id_set:
            if instance[ID] > 10 and instance[ID] < 20:
                all_ids.add(ID)

    clsDic = {}

    for i in all_ids:
        clsDic[i] = len(clsDic)
    # print(len(clsDic))
    with open(os.path.join(root_dir, 'instanceID_2.json'), 'w') as f:
        json.dump(clsDic, f)

def createInstanceID_ALL(root_dir='data'):
    modes = ['train', 'validation_2', 'validation']

    all_ids = set([])
    for mode in modes:
        img_tat = mode + '_images'
        vdo_tat = mode + '_videos'

        with open(os.path.join(root_dir, img_tat+'_annotation.json'), 'r') as f:
            d_i = json.load(f)
        with open(os.path.join(root_dir, vdo_tat+'_annotation.json'), 'r') as f:
            d_v = json.load(f)

        l_i = d_i['annotations']
        l_v = d_v['annotations']

        instance = {}
        s_i = set([])
        s_v = set([])

        for d in l_i:
            for dd in d['annotations']:
                if dd['instance_id'] > 0:
                    s_i.add(dd['instance_id'])
                    if dd['instance_id'] not in instance:
                        instance[dd['instance_id']] = 1
                    else:
                        instance[dd['instance_id']] += 1

        for d in l_v:
            for dd in d['annotations']:
                if dd['instance_id'] > 0:
                    s_v.add(dd['instance_id'])
                    if dd['instance_id'] not in instance:
                        instance[dd['instance_id']] = 1
                    else:
                        instance[dd['instance_id']] += 1

        id_set = s_i & s_v
        
        for ID in id_set:
            if instance[ID] > 10 and instance[ID] < 20:
                all_ids.add(ID)

    clsDic = {}

    for i in all_ids:
        clsDic[i] = len(clsDic)
    # print(len(clsDic))
    with open(os.path.join(root_dir, 'instanceID_all.json'), 'w') as f:
        json.dump(clsDic, f)

def createText(mode, root_dir='data'):
    if mode == 'train':
        roots = [os.path.join(root_dir, 'train_dataset_part{}'.format(i+1)) for i in range(6)]
    elif mode == 'validation':
        roots = [os.path.join(root_dir, 'validation_dataset_part{}'.format(i+1)) for i in range(2)]
    elif mode == 'validation_2':
        roots = [os.path.join(root_dir, 'validation_dataset_part{}'.format(i+3)) for i in range(2)]

    img_dic = {}
    vdo_dic = {}

    for root in tqdm(roots):
        img_root = os.path.join(root, 'image_text')
        vdo_root = os.path.join(root, 'video_text')

        imgs = os.listdir(img_root)
        for img_f in imgs:
            with open(os.path.join(img_root, img_f)) as f:
                text = f.readline()
            img_dic[img_f[:-4]] = text
        
        vdos = os.listdir(vdo_root)
        for vdo_f in vdos:
            with open(os.path.join(vdo_root, vdo_f)) as f:
                text = f.readline()
            vdo_dic[vdo_f[:-4]] = text

    with open(os.path.join(root_dir, '{}_images_text.json'.format(mode)), 'w') as f:
        json.dump(img_dic, f)
    with open(os.path.join(root_dir, '{}_videos_text.json'.format(mode)), 'w') as f:
        json.dump(vdo_dic, f)

def createVocab(root_dir='data', mode='train'):
    with open(os.path.join(root_dir, '{}_images_text.json'.format(mode)), 'r') as f:
        img_dic = json.load(f)
    with open(os.path.join(root_dir, '{}_videos_text.json'.format(mode)), 'r') as f:
        vdo_dic = json.load(f)

    with open(os.path.join(root_dir, 'stop_use.txt'), 'r') as f:
        stop_use = f.readlines()
    stop_use = [w.strip() for w in stop_use]
    
    allwords = []
    for value in tqdm(img_dic.values()):
        words = jieba.cut(value, cut_all=False, HMM=True)
        words = [w.strip() for w in words if w.strip() not in stop_use and w.strip()]
        allwords += words

    for value in tqdm(vdo_dic.values()):
        words = jieba.cut(value, cut_all=False, HMM=True)
        words = [w.strip() for w in words if w.strip() not in stop_use and w.strip()]
        allwords += words

    allwords = set(allwords)

    vocab = {}
    vocab['[PAD]'] = 0

    for i, w in enumerate(allwords):
        vocab[w] = i+1
    
    with open(os.path.join(root_dir, 'vocab.json'), 'w') as f:
        json.dump(vocab, f)

def createTF_IDF(root_dir='data', mode='train'):
    with open(os.path.join(root_dir, '{}_images_text.json'.format(mode)), 'r') as f:
        img_dic = json.load(f)
    with open(os.path.join(root_dir, '{}_videos_text.json'.format(mode)), 'r') as f:
        vdo_dic = json.load(f)

    with open(os.path.join(root_dir, 'vocab.json'), 'r') as f:
        vocab = json.load(f)

    with open(os.path.join(root_dir, 'stop_use.txt'), 'r') as f:
        stop_use = f.readlines()
    stop_use = [w.strip() for w in stop_use]
    
    TF = {}
    IDF = {}
    for value in tqdm(img_dic.values()):
        words = jieba.cut(value, cut_all=False, HMM=True)
        words = [w.strip() for w in words if w.strip() not in stop_use and w.strip()]
        for w in words:
            if w in TF:
                TF[w] += 1
            else:
                TF[w] = 1
        for w in set(words):
            if w in IDF:
                IDF[w] += 1
            else:
                IDF[w] = 1

    for value in tqdm(vdo_dic.values()):
        words = jieba.cut(value, cut_all=False, HMM=True)
        words = [w.strip() for w in words if w.strip() not in stop_use and w.strip()]
        for w in words:
            if w in TF:
                TF[w] += 1
            else:
                TF[w] = 1
        for w in set(words):
            if w in IDF:
                IDF[w] += 1
            else:
                IDF[w] = 1

    s = sum(TF.values())
    for k in TF.keys():
        TF[k] /= s
    
    s = len(img_dic) + len(vdo_dic)
    for k in IDF.keys():
        IDF[k] = np.log10(s/IDF[k])

    dic = {}
    for w in IDF.keys():
        dic[vocab[w]] = TF[w]*IDF[w]
    dic[vocab['[PAD]']] = 0
    with open(os.path.join(root_dir, 'TF_IDF.json'), 'w') as f:
        json.dump(dic, f)
    

if __name__ == "__main__":
    label = {}
    label['label2index'] = {}
    label['index2label'] = {}
    processTrain(label)
    processValidation(label)
    processValidation_2(label)
    saveNumpyInstance('data', 'train', (270, 270))
    saveNumpyInstance('data', 'validation', (270, 270))
    saveNumpyInstance('data', 'validation_2', (270, 270))
    createInstance2Label('data')
    # createInstanceID()
    # createInstanceID_ALL()
    # createText(mode='train')
    # createText(mode='validation')
    # createText(mode='validation_2')
    # createVocab()
    # createTF_IDF()

