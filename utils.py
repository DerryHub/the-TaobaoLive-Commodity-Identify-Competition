import os
import torch
from torch.optim.optimizer import Optimizer
from torch.functional import F
import numpy as np
import math
import cv2

colors = [(39, 129, 113), (164, 80, 133), (83, 122, 114), (99, 81, 172), (95, 56, 104), (37, 84, 86), (14, 89, 122),
          (80, 7, 65), (10, 102, 25), (90, 185, 109), (106, 110, 132), (169, 158, 85), (188, 185, 26), (103, 1, 17),
          (82, 144, 81), (92, 7, 184), (49, 81, 155), (179, 177, 69), (93, 187, 158), (13, 39, 73), (12, 50, 60),
          (16, 179, 33), (112, 69, 165), (15, 139, 63), (33, 191, 159), (182, 173, 32), (34, 113, 133), (90, 135, 34),
          (53, 34, 86), (141, 35, 190), (6, 171, 8), (118, 76, 112), (89, 60, 55), (15, 54, 88), (112, 75, 181),
          (42, 147, 38), (138, 52, 63), (128, 65, 149), (106, 103, 24), (168, 33, 45), (28, 136, 135), (86, 91, 108),
          (52, 11, 76), (142, 6, 189), (57, 81, 168), (55, 19, 148), (182, 101, 89), (44, 65, 179), (1, 33, 26),
          (122, 164, 26), (70, 63, 134), (137, 106, 82), (120, 118, 52), (129, 74, 42), (182, 147, 112), (22, 157, 50),
          (56, 50, 20), (2, 22, 177), (156, 100, 106), (21, 35, 42), (13, 8, 121), (142, 92, 28), (45, 118, 33),
          (105, 118, 30), (7, 185, 124), (46, 34, 146), (105, 184, 169), (22, 18, 5), (147, 71, 73), (181, 64, 91),
          (31, 39, 184), (164, 179, 33), (96, 50, 18), (95, 15, 106), (113, 68, 54), (136, 116, 112), (119, 139, 130),
          (31, 139, 34), (66, 6, 127), (62, 39, 2), (49, 99, 180), (49, 119, 155), (153, 50, 183), (125, 38, 3),
          (129, 87, 143), (49, 87, 40), (128, 62, 120), (73, 85, 148), (28, 144, 118), (29, 9, 24), (175, 45, 108),
          (81, 175, 64), (178, 19, 157), (74, 188, 190), (18, 114, 2), (62, 128, 96), (21, 3, 150), (0, 6, 95),
          (2, 20, 184), (122, 37, 185)]

class MSE_match:
    def __call__(self, f1, f2):
        return torch.mean(torch.sum((f1-f2)**2, dim=1))

def collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]
    text = [s['text'] for s in data]

    imgs = torch.from_numpy(np.stack(imgs, axis=0))
    text = torch.stack(text, dim=0)

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 6)) * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 6)) * -1

    imgs = imgs.permute(0, 3, 1, 2)

    return {'img': imgs, 'annot': annot_padded, 'scale': scales, 'text': text}


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample, common_size=512):
        image, annots, text = sample['img'], sample['annot'], sample['text']
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
        return {
            'img': torch.from_numpy(new_image), 
            'annot': torch.from_numpy(annots), 
            'scale': scale, 
            'text': text
        }


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):
        if np.random.rand() < flip_x:
            image, annots, text = sample['img'], sample['annot'], sample['text']
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            sample = {'img': image, 'annot': annots, 'text': text}

        return sample


class Normalizer(object):

    def __init__(self):
        self.mean = np.array([[[0.64364545, 0.60998588, 0.60550367]]])
        self.std = np.array([[[0.22700769, 0.23887326, 0.23833767]]])

    def __call__(self, sample):
        image, annots, text = sample['img'], sample['annot'], sample['text']

        return {'img': ((image.astype(np.float32) - self.mean) / self.std), 'annot': annots, 'text': text}

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

def area(boxs):
    h = boxs[:, 3] - boxs[:, 1]
    w = boxs[:, 2] - boxs[:, 0]
    area = w * h
    return area

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

def collater_HardTriplet(data):
    imgs = [s['img'] for s in data]
    instances = [s['instance'] for s in data]

    imgs = torch.cat(imgs, dim=0)
    instances = torch.cat(instances, dim=0)

    return {'img': imgs, 'instance': instances}

class TripletFocalLoss():
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
        return distance
    
class TripletLoss():
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, feature_q, feature_p, feature_n):
        distance_p = self.distance(feature_q, feature_p)
        distance_n = self.distance(feature_q, feature_n)
        loss = F.relu(distance_p-distance_n+self.threshold)
        loss = torch.mean(loss)
        return loss

    def distance(self, f_1, f_2):
        distance = torch.sum((f_1-f_2)**2, dim=1) / 4
        return distance

def hard_example_mining(dist_mat, labels, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
        dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
        labels: pytorch LongTensor, with shape [N]
        return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
        dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
        dist_an: pytorch Variable, distance(anchor, negative); shape [N]
        p_inds: pytorch LongTensor, with shape [N]; 
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
        n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples, 
        thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape
    dist_ap, relative_p_inds = torch.max(dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an, relative_n_inds = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    # shape [N]

    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels)
            .copy_(torch.arange(0, N).long())
            .unsqueeze( 0).expand(N, N))
        # shape [N, 1]
        p_inds = torch.gather(
        ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
        ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an

class HardTripletLoss:
    def __init__(self, threshold):
        self.threshold = threshold
    
    def __call__(self, features, instances):
        dis = 1 - torch.mm(features, features.t())
        dist_ap, dist_an = hard_example_mining(dis, instances)
        loss = F.relu(dist_ap-dist_an+self.threshold)
        loss = torch.mean(loss)
        # argmin = torch.argmin(dis, dim=1)
        _, argmin = torch.topk(dis, 2, dim=1, largest=False)
        argmin_2 = argmin[:, -1]
        acc = (instances[argmin_2]==instances).sum().float()
        return loss, acc

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

def separate_bn_paras(modules):
    if not isinstance(modules, list):
        modules = [*modules.modules()]
    paras_only_bn = []
    paras_wo_bn = []
    
    for layer in modules:
        if 'arcface' in str(layer.__class__):
            continue
        if 'container' in str(layer.__class__):
            continue
        else:
            if 'batchnorm' in str(layer.__class__):
                paras_only_bn.extend([*layer.parameters()])
            else:
                paras_wo_bn.extend([*layer.parameters()])
    return paras_only_bn, paras_wo_bn

class AdamW(Optimizer):
    r"""Implements AdamW algorithm.
    The original Adam algorithm was proposed in `Adam: A Method for Stochastic Optimization`_.
    The AdamW variant was proposed in `Decoupled Weight Decay Regularization`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(AdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Perform stepweight decay
                p.mul_(1 - group['lr'] * group['weight_decay'])

                # Perform optimization step
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1

                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

if __name__ == "__main__":
    # a = [[1,3,4,5]]*10+[[1,3,6,7]]
    # a = np.array(a)
    # print(area(a))
    # print(a[np.argsort(-area(a))])
    cost = MSE_match()
    a = torch.tensor([[1,2.],[4,5]])
    b = torch.tensor([[2,2.],[4,5]])
    print(cost(a,b))