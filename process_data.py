import os
import json
from shutil import copyfile, rmtree
from tqdm import tqdm
import cv2

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

def processTrain():
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

    label = {}
    label['label2index'] = {}
    label['index2label'] = {}

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

def processValidation():
    roots = [1, 2]
    img_tat = 'data/validation_images'
    vdo_tat = 'data/validation_videos' 

    if os.path.isdir(img_tat):
        rmtree(img_tat)
    os.makedirs(img_tat)

    if os.path.isdir(vdo_tat):
        rmtree(vdo_tat)
    os.makedirs(vdo_tat)

    label = {}
    label['label2index'] = {}
    label['index2label'] = {}

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

if __name__ == "__main__":
    processTrain()
    processValidation()

    

