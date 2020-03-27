import os
import torch
import numpy as np
import cv2

'''
    for test
'''

class Normalizer_Test(object):

    def __init__(self):
        self.mean = np.array([[[0.64364545, 0.60998588, 0.60550367]]])
        self.std = np.array([[[0.22700769, 0.23887326, 0.23833767]]])

    def __call__(self, sample):
        image, text = sample['img'], sample['text']

        return {'img': ((image.astype(np.float32) - self.mean) / self.std), 'text': text}


class Resizer_Test(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample, common_size=512):
        image, text = sample['img'], sample['text']
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

        return {'img': torch.from_numpy(new_image), 'scale': scale, 'text': text}

def collater_test(data):
    imgs = [s['img'] for s in data]
    scales = [s['scale'] for s in data]
    text = [s['text'] for s in data]

    imgs = torch.from_numpy(np.stack(imgs, axis=0))
    text = torch.stack(text, dim=0)

    imgs = imgs.permute(0, 3, 1, 2)

    return {'img': imgs, 'scale': scales, 'text': text}