from torch import nn, Tensor
import torch
import math
import torch.nn.functional as F
from arcface.utils import l2_norm

class Arcface(nn.Module):
    # implementation of additive margin softmax loss in https://arxiv.org/abs/1801.05599    
    def __init__(self, config):
        super(Arcface, self).__init__()
        embedding_size = config.embedding_size
        num_classes = config.num_classes
        s = config.s
        m = config.m
        # m1 = config.m1
        # m2 = config.m2
        # m3 = config.m3

        self.num_classes = num_classes
        self.kernel = nn.Parameter(torch.Tensor(embedding_size, num_classes))
        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = m # the margin value, default is 0.5
        self.s = s # scalar value default is 64, see normface https://arxiv.org/abs/1704.06369
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.mm = self.sin_m * m  # issue 1
        self.threshold = math.cos(math.pi - m)


    def forward(self, inputs):
        if self.training:
            embbedings, label = inputs
        else:
            embbedings = inputs
            label = None
        # weights norm
        nB = len(embbedings)
        kernel_norm = l2_norm(self.kernel,axis=0)
        # cos(theta+m)
        cos_theta = torch.mm(embbedings, kernel_norm)
#         output = torch.mm(embbedings,kernel_norm)
        cos_theta = cos_theta.clamp(-1,1) # for numerical stability
        cos_theta_2 = torch.pow(cos_theta, 2)
        sin_theta_2 = 1 - cos_theta_2
        sin_theta = torch.sqrt(sin_theta_2)
        cos_theta_m = (cos_theta * self.cos_m - sin_theta * self.sin_m)
        # this condition controls the theta+m should in range [0, pi]
        #      0<=theta+m<=pi
        #     -m<=theta<=pi-m
        cond_v = cos_theta - self.threshold
        cond_mask = cond_v <= 0
        keep_val = (cos_theta - self.mm) # when theta not in [0,pi], use cosface instead
        cos_theta_m[cond_mask] = keep_val[cond_mask]
        output = cos_theta * 1.0 # a little bit hacky way to prevent in_place operation on cos_theta
        idx_ = torch.arange(0, nB, dtype=torch.long)
        if label is not None:
            output[idx_, label] = cos_theta_m[idx_, label]
        output *= self.s # scale up in order to make softmax work, first introduced in normface
        return output

class AdaCos(nn.Module):
    def __init__(self, config):
        super(AdaCos, self).__init__()
        num_features = config.embedding_size
        num_classes = config.num_classes
        self.m = config.m
        self.num_features = num_features
        self.n_classes = num_classes
        self.s = math.sqrt(2) * math.log(num_classes - 1)
        self.kernel = nn.Parameter(torch.FloatTensor(num_classes, num_features))
        nn.init.xavier_uniform_(self.kernel)

    def forward(self, inputs):
        if self.training:
            input, label = inputs
        else:
            input = inputs
            label = None
        # normalize features
        x = F.normalize(input)
        # normalize weights
        W = F.normalize(self.kernel)
        # dot product
        logits = F.linear(x, W)
        if label is None:
            return logits
        # feature re-scale
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        with torch.no_grad():
            B_avg = torch.where(one_hot < 1, torch.exp(self.s * logits), torch.zeros_like(logits))
            B_avg = torch.sum(B_avg) / input.size(0)
            # print(B_avg)
            theta_med = torch.median(theta[one_hot == 1])
            self.s = torch.log(B_avg) / torch.cos(torch.min(math.pi/4 * torch.ones_like(theta_med), theta_med))
        output = self.s * logits

        return output

class SparseCircleLoss(nn.Module):
    def __init__(self, config) -> None:
        super(SparseCircleLoss, self).__init__()
        self.margin = config.m
        self.gamma = config.s
        self.soft_plus = nn.Softplus()
        self.class_num = config.num_classes
        self.emdsize = config.embedding_size

        self.kernel = nn.Parameter(torch.FloatTensor(self.class_num, self.emdsize))
        nn.init.xavier_uniform_(self.kernel)


    def forward(self, inputs) -> Tensor:
        input, label = inputs
        similarity_matrix = nn.functional.linear(nn.functional.normalize(input,p=2, dim=1, eps=1e-12), nn.functional.normalize(self.kernel,p=2, dim=1, eps=1e-12))
        
        one_hot = torch.zeros(similarity_matrix.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        one_hot = one_hot.type(dtype=torch.bool)
        #sp = torch.gather(similarity_matrix, dim=1, index=label.unsqueeze(1))
        sp = similarity_matrix[one_hot]
        # mask = one_hot.logical_not()
        mask = torch.logical_not(one_hot)
        sn = similarity_matrix[mask]

        sp = sp.view(input.size()[0], -1)
        sn = sn.view(input.size()[0], -1)

        ap = torch.clamp_min(-sp.detach() + 1 + self.margin, min=0.)
        an = torch.clamp_min(sn.detach() + self.margin, min=0.)

        delta_p = 1 - self.margin
        delta_n = self.margin

        logit_p = - ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        loss = self.soft_plus(torch.logsumexp(logit_n, dim=1) + torch.logsumexp(logit_p, dim=1))

        return loss.mean(), similarity_matrix

class LinearLayer(nn.Module):
    def __init__(self, config):
        super(LinearLayer, self).__init__()
        embedding_size = config.embedding_size
        num_labels = config.num_labels
        self.fc = nn.Linear(embedding_size, num_labels)
    
    def forward(self, embbedings):
        output = self.fc(embbedings)
        return output
