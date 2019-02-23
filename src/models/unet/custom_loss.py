"""
Lovasz-Softmax and Jaccard hinge loss in PyTorch
Maxim Berman 2018 ESAT-PSI KU Leuven (MIT License)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(pred, target):
    smooth = 1.

    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()

    return 1 - ((2. * intersection + smooth) / (pred.sum() + target.sum() + smooth))

def lovasz_grad(gt_sorted):
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * Variable(signs))
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    return loss
def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = mean(lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), 
            lab.unsqueeze(0), ignore))
                          for log, lab in zip(logits, labels))
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss

def isnan(x):
    return x != x
    
    
def mean(l, ignore_nan=True, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n




from .transforms import convert_labels_to_one_hot_encoding

# TODO: version of pytorch for cuda 7.5 doesn't have the latest features like
# reduce=False argument -- update cuda on the machine and update the code

# TODO: update the class to inherit the nn.Weighted loss with all the additional
# arguments

class FocalLoss(nn.Module):
    """Focal loss puts more weight on more complicated examples."""
   
    def __init__(self, gamma=1):
        
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, flatten_logits, flatten_targets):
        
        flatten_targets = flatten_targets.data
        
        number_of_classes = flatten_logits.size(1)
        
        flatten_targets_one_hot = convert_labels_to_one_hot_encoding(flatten_targets, number_of_classes)

        all_class_probabilities = F.softmax(flatten_logits)

        probabilities_of_target_classes = all_class_probabilities[flatten_targets_one_hot]

        elementwise_loss =  - (1 - probabilities_of_target_classes).pow(self.gamma) * torch.log(probabilities_of_target_classes)
        
        return elementwise_loss.sum()
