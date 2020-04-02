import torch
import torch.nn as nn


def calc_iou(a, b):

    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])
    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)
    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih
    ua = torch.clamp(ua, min=1e-8)
    intersection = iw * ih
    IoU = intersection / ua

    return IoU


class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()

    def forward(self, instances, classifications, regressions, anchors, annotations):
        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]
        instance_losses = []
        classification_losses = []
        regression_losses = []

        anchor = anchors[0, :, :]

        anchor_widths = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x = anchor[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y = anchor[:, 1] + 0.5 * anchor_heights

        for j in range(batch_size):
            
            instance = instances[j, :, :]
            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            bbox_annotation = annotations[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

            if bbox_annotation.shape[0] == 0:
                if torch.cuda.is_available():
                    regression_losses.append(torch.tensor(0).float().cuda())
                    classification_losses.append(torch.tensor(0).float().cuda())
                else:
                    regression_losses.append(torch.tensor(0).float())
                    classification_losses.append(torch.tensor(0).float())

                continue
            
            instance = torch.clamp(instance, 1e-4, 1.0 - 1e-4)
            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

            IoU = calc_iou(anchors[0, :, :], bbox_annotation[:, :4])

            IoU_max, IoU_argmax = torch.max(IoU, dim=1)

            # compute the loss for classification
            targets_instance = torch.ones(instance.shape) * -1
            targets = torch.ones(classification.shape) * -1
            if torch.cuda.is_available():
                targets = targets.cuda()
                targets_instance = targets_instance.cuda()

            targets[torch.lt(IoU_max, 0.4), :] = 0
            # targets_instance[torch.lt(IoU_max, 0.4), :] = 0

            positive_indices = torch.ge(IoU_max, 0.5)
            positive_indices_instance = torch.ge(IoU_max, 0.3)
            # print(positive_indices.size(), targets.size())
            # print(bbox_annotation.size())

            num_positive_anchors = positive_indices.sum()
            num_positive_anchors_instance = positive_indices_instance.sum()

            assigned_annotations = bbox_annotation[IoU_argmax, :]

            # print(positive_indices_instance.size())
            # print((positive_indices & (assigned_annotations[:, 5] == 1)).size())
            targets_instance[positive_indices_instance & (assigned_annotations[:, 5] == 1), :] = 1
            targets_instance[positive_indices_instance & (assigned_annotations[:, 5] == 0), :] = 0
            # print(num_positive_anchors, (positive_indices & (assigned_annotations[:, 5] == 1)).size())

            targets[positive_indices, :] = 0
            targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1

            alpha_factor = torch.ones(targets.shape) * alpha
            alpha_factor_instance = torch.ones(targets_instance.shape) * alpha
            if torch.cuda.is_available():
                alpha_factor = alpha_factor.cuda()
                alpha_factor_instance = alpha_factor_instance.cuda()

            # 正负例权重
            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            alpha_factor_instance = torch.where(torch.eq(targets_instance, 1.), alpha_factor_instance, 1. - alpha_factor_instance)
            # focal loss
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            focal_weight_instance = torch.where(torch.eq(targets_instance, 1.), 1. - instance, instance)
            focal_weight_instance = alpha_factor_instance * torch.pow(focal_weight_instance, gamma)

            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))
            bce_instance = -(targets_instance * torch.log(instance) + (1.0 - targets_instance) * torch.log(1.0 - instance))

            cls_loss = focal_weight * bce
            instance_loss = focal_weight_instance * bce_instance

            zeros = torch.zeros(cls_loss.shape)
            zeros_instance = torch.zeros(instance_loss.shape)
            if torch.cuda.is_available():
                zeros = zeros.cuda()
                zeros_instance = zeros_instance.cuda()
            # 对不是正例也不是负例的loss筛选
            cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, zeros)
            instance_loss = torch.where(torch.ne(targets_instance, -1.0), instance_loss, zeros_instance)

            classification_losses.append(cls_loss.sum() / torch.clamp(num_positive_anchors.float(), min=1.0))
            instance_losses.append(instance_loss.sum() / torch.clamp(num_positive_anchors_instance.float(), min=1.0))

            if positive_indices.sum() > 0:
                # 正样本对应的annotations
                assigned_annotations = assigned_annotations[positive_indices, :]
                # print(assigned_annotations.size())
                # IoU>0.5对应的anchor
                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]
                
                gt_widths = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y = assigned_annotations[:, 1] + 0.5 * gt_heights

                gt_widths = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)

                targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
                targets = targets.t()

                norm = torch.Tensor([[0.1, 0.1, 0.2, 0.2]])
                if torch.cuda.is_available():
                    norm = norm.cuda()
                targets = targets / norm

                regression_diff = torch.abs(targets - regression[positive_indices, :])

                regression_loss = torch.where(
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                )
                regression_losses.append(regression_loss.mean())
            else:
                if torch.cuda.is_available():
                    regression_losses.append(torch.tensor(0).float().cuda())
                else:
                    regression_losses.append(torch.tensor(0).float())

        return torch.stack(instance_losses).mean(dim=0, keepdim=True), torch.stack(classification_losses).mean(dim=0, keepdim=True), torch.stack(regression_losses).mean(dim=0,
                                                                                                                 keepdim=True)
if __name__ == "__main__":
    a = torch.Tensor([[1,1,2,2],[2,2,3,3]])
    b = torch.Tensor([[1,1,2,2],[1,1,2,2]])
    print(calc_iou(a,b))