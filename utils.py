import os
import torch
import numpy as np
import cv2

def collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]

    imgs = torch.from_numpy(np.stack(imgs, axis=0))

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    imgs = imgs.permute(0, 3, 1, 2)

    return {'img': imgs, 'annot': annot_padded, 'scale': scales}


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample, common_size=512):
        image, annots = sample['img'], sample['annot']
        height, width, _ = image.shape
        if height > width:
            scale = common_size / height
            resized_height = common_size
            resized_width = int(width * scale)
        else:
            scale = common_size / width
            resized_height = int(height * scale)
            resized_width = common_size

        image = cv2.resize(image, (resized_width, resized_height))

        new_image = np.zeros((common_size, common_size, 3))
        new_image[0:resized_height, 0:resized_width] = image

        annots[:, :4] *= scale

        return {'img': torch.from_numpy(new_image), 'annot': torch.from_numpy(annots), 'scale': scale}


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):
        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            sample = {'img': image, 'annot': annots}

        return sample


class Normalizer(object):

    def __init__(self):
        self.mean = np.array([[[0.64364545, 0.60998588, 0.60550367]]])
        self.std = np.array([[[0.22700769, 0.23887326, 0.23833767]]])

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']

        return {'img': ((image.astype(np.float32) - self.mean) / self.std), 'annot': annots}

def iou(a, b):
    a = torch.clamp(a.long(), 0, 2000)
    b = torch.clamp(b.long(), 0, 2000)
    img_a = torch.zeros([2000, 2000])
    img_b = torch.zeros([2000, 2000])
    for t in a:
        img_a[t[0]:t[2], t[1]:t[3]] = 1
    for t in b:
        img_b[t[0]:t[2], t[1]:t[3]] = 1
    intersection = img_a*img_b
    ua = torch.clamp(img_a+img_b, max=1)
    return (intersection.sum()+1e-8) / (ua.sum()+1e-8)

'''
    for test
'''

class Normalizer_Test(object):

    def __init__(self):
        self.mean = np.array([[[0.64364545, 0.60998588, 0.60550367]]])
        self.std = np.array([[[0.22700769, 0.23887326, 0.23833767]]])

    def __call__(self, sample):
        image = sample['img']

        return {'img': ((image.astype(np.float32) - self.mean) / self.std)}


class Resizer_Test(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample, common_size=512):
        image = sample['img']
        height, width, _ = image.shape
        if height > width:
            scale = common_size / height
            resized_height = common_size
            resized_width = int(width * scale)
        else:
            scale = common_size / width
            resized_height = int(height * scale)
            resized_width = common_size

        image = cv2.resize(image, (resized_width, resized_height))

        new_image = np.zeros((common_size, common_size, 3))
        new_image[0:resized_height, 0:resized_width] = image

        return {'img': torch.from_numpy(new_image), 'scale': scale}

def collater_test(data):
    imgs = [s['img'] for s in data]
    scales = [s['scale'] for s in data]

    imgs = torch.from_numpy(np.stack(imgs, axis=0))

    imgs = imgs.permute(0, 3, 1, 2)

    return {'img': imgs, 'scale': scales}

class TripletLoss():
    def __init__(self, alpha, gamma):
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, feature_q, feature_p, feature_n):
        distance_p = self.distance(feature_q, feature_p)
        loss_p = torch.mean(-torch.log(1-distance_p+1e-8)*(distance_p**self.gamma)*self.alpha)
        distance_n = self.distance(feature_q, feature_n)
        loss_n = torch.mean(-torch.log(distance_n+1e-8)*((1-distance_n)**self.gamma)*(1-self.alpha))
        return loss_p + loss_n

    def distance(self, f_1, f_2):
        distance = torch.sum((f_1-f_2)**2, dim=1) / 4
        # distance = torch.mean(distance)
        return distance
    
class TripletAccuracy():
    def __call__(self, feature_q, feature_p, feature_n):
        distance_p = self.distance(feature_q, feature_p)
        distance_n = self.distance(feature_q, feature_n)
        total_p = distance_p.size(0) 
        total_n = distance_n.size(0)
        acc_p = torch.sum(distance_p < 0.5).float()
        acc_n = torch.sum(distance_n > 0.5).float()
        return acc_p, acc_n, total_p, total_n

    def distance(self, f_1, f_2):
        distance = torch.sum((f_1-f_2)**2, dim=1) / 4
        return distance


if __name__ == "__main__":
    cost = TripletLoss(0.25, 1.5)
    a = torch.tensor([[1,0.],[0.8,0.6]])
    b = torch.tensor([[1,0.],[0.8,0.6]])
    c = torch.tensor([[-1,0], [1,0]])
    print(cost(a,b,c))