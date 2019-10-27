import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable


def predict(X, threshold):
    '''X is sigmoid output of the model'''
    X_p = np.copy(X)
    preds = (X_p > threshold).astype('uint8')
    return preds

def metric(probability, truth, threshold=0.5, reduction='none'):
    '''Calculates dice of positive and negative images seperately'''
    '''probability and truth must be torch tensors'''
    batch_size = len(truth)
    with torch.no_grad():
        probability = probability.view(batch_size, -1)
        truth = truth.view(batch_size, -1)
        assert(probability.shape == truth.shape)

        p = (probability > threshold).float()
        t = (truth > 0.5).float()

        t_sum = t.sum(-1)
        p_sum = p.sum(-1)
        neg_index = torch.nonzero(t_sum == 0)
        pos_index = torch.nonzero(t_sum >= 1)

        dice_neg = (p_sum == 0).float()
        dice_pos = 2 * (p*t).sum(-1)/((p+t).sum(-1))

        dice_neg = dice_neg[neg_index]
        dice_pos = dice_pos[pos_index]
        dice = torch.cat([dice_pos, dice_neg])

        dice_neg = np.nan_to_num(dice_neg.mean().item(), 0)
        dice_pos = np.nan_to_num(dice_pos.mean().item(), 0)
        dice = dice.mean().item()

        num_neg = len(neg_index)
        num_pos = len(pos_index)

    return dice, dice_neg, dice_pos, num_neg, num_pos


def dice_loss(pred,mask):
    ''' flatten version:
    batch_size = pred.shape[0]
    smooth = 1
    pred = pred.view(batch_size,-1)
    mask = mask.view(batch_size,-1)
    score = (2. * (pred*mask).sum(-1)+smooth) / ((pred+mask).sum(-1)+smooth)

    return (1-score).mean()'''

    #4 classes version
    smooth = 1
    dice = 2*((pred*mask).sum(-1).sum(-1)+smooth) / ((pred+mask).sum(-1).sum(-1)+smooth)
    loss = 1-dice.mean()
    #assert loss >= 0, loss
    return loss


def generalized_dice_loss(pred,refe):
    '''r_ln = refe.sum(-1).sum(-1) #[n,c]
    smooth = 1e-6
    w_l = 1 / (r_ln**2 + smooth) #[n,c]

    num = ( w_l*((pred*refe).sum(-1).sum(-1)) ).sum(-1) #[n]
    den = ( w_l*((pred+refe).sum(-1).sum(-1)) ).sum(-1) #[n]
    loss = 1-2*num/den
    return loss.mean()'''

    def flatten(tensor):
        """Flattens a given tensor such that the channel axis is first.
        The shapes are transformed as follows:
           (N, C, H, W) -> (C, N * H * W)
        """
        C = tensor.size(1)
        axis_order = (1, 0) + tuple(range(2, tensor.dim()))
        transposed = tensor.permute(axis_order).contiguous()
        return transposed.view(C, -1)

    smooth = 1e-5
    input = flatten(pred)
    target = flatten(refe)
    target = target.float()
    target_sum = target.sum(-1)
    class_weights = Variable(1. / (target_sum * target_sum).clamp(min=smooth), requires_grad=False)

    intersect = (input * target).sum(-1) * class_weights
    denominator = ((input + target).sum(-1) * class_weights)

    return (1. - 2. * intersect / denominator.clamp(min=smooth)).mean()



def dice_LB(pred,mask,threshold):
    #both are torch.Tensor
    #pred:after sigmoid
    '''
        assert pred.shape == mask.shape
        batch_size = pred.shape[0]
        pred = pred > threshold
        sigma = 1
        pred_layer = pred.view(batch_size,-1).float()
        mask_layer = mask.view(batch_size,-1).float()
        dice = (2 * (pred_layer * mask_layer).sum() + sigma) / (pred_layer.sum() + mask_layer.sum() + sigma)
        return dice.mean()'''

    pred = pred.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy()

    assert pred.shape == mask.shape
    num_classes = pred.shape[1]
    batch_size = pred.shape[0]
    pred = pred>threshold#torch.where(pred>threshold,1,0)
    dice = 0
    sigma = 1
    for b in range(batch_size):
        for c in range(num_classes):
            pred_layer = pred[b][c].astype(np.float32)
            mask_layer = mask[b][c].astype(np.float32)
            dice+=\
                (2*(pred_layer*mask_layer).sum()+sigma) / (pred_layer.sum() + mask_layer.sum()+sigma)

    return dice/(num_classes*batch_size)



class Meter:
    '''A meter to keep track of iou and dice scores throughout an epoch'''
    def __init__(self, phase, epoch):
        self.base_threshold = 0.5 # <<<<<<<<<<< here's the threshold
        self.base_dice_scores = []
        self.dice_neg_scores = []
        self.dice_pos_scores = []
        self.iou_scores = []

    def update(self, targets, outputs):
        _, dice_neg, dice_pos, _, _ = metric(outputs, targets, self.base_threshold)
        dice = dice_LB(outputs,targets,self.base_threshold)
        #dice_neg = dice_pos = 0
        self.base_dice_scores.append(dice)
        self.dice_pos_scores.append(dice_pos)
        self.dice_neg_scores.append(dice_neg)
        preds = predict(outputs, self.base_threshold)
        iou = compute_iou_batch(preds, targets, classes=[1])
        self.iou_scores.append(iou)

    def get_metrics(self):
        dice = np.mean(self.base_dice_scores)
        dice_neg = np.mean(self.dice_neg_scores)
        dice_pos = np.mean(self.dice_pos_scores)
        dices = [dice, dice_neg, dice_pos]
        iou = np.nanmean(self.iou_scores)
        return dices, iou

def epoch_log(epoch, phase, file_name, epoch_loss, meter):
    '''logging the metrics at the end of an epoch'''
    dices, iou = meter.get_metrics()
    dice, dice_neg, dice_pos = dices
    info = "Loss: %0.4f | IoU: %0.4f | dice: %0.4f | dice_neg: %0.4f | dice_pos: %0.4f" % \
           (epoch_loss, iou, dice, dice_neg, dice_pos)
    f = open(file_name, 'a')
    if phase == 'train':
        f.write('epoch:' + str(epoch) + '\n')
    f.write(info + '\n')
    if phase == 'val':
        f.write('\n')
    f.close()
    print(info)

    return dice, iou

def compute_ious(pred, label, classes, ignore_index=255, only_present=True):
    '''computes iou for one ground truth mask and predicted mask'''
    pred[label == ignore_index] = 0
    ious = []
    for c in classes:
        label_c = label == c
        if only_present and np.sum(label_c) == 0:
            ious.append(np.nan)
            continue
        pred_c = pred == c
        intersection = np.logical_and(pred_c, label_c).sum()
        union = np.logical_or(pred_c, label_c).sum()
        if union != 0:
            ious.append(intersection / union)
    return ious if ious else [1]

def compute_iou_batch(outputs, labels, classes):
    '''computes mean iou for a batch of ground truth masks and predicted masks'''
    ious = []
    preds = np.copy(outputs) # copy is imp
    labels = np.array(labels) # tensor to np
    for pred, label in zip(preds, labels):
        ious.append(np.nanmean(compute_ious(pred, label, classes)))
    iou = np.nanmean(ious)
    return iou

def weighted_BCE_loss(p_hat,p):
    #p_hat:prediction   p:ground-truth
    #beta: weight between bg and fg
    #p_hat is calculated after sigmoid
    smooth = 1e-3
    beta = 50
    loss = - beta * p * torch.log(p_hat+smooth) - (1-p)*torch.log(1-p_hat+smooth)
    loss = loss.mean()
    return loss


def binary_focal_loss(p_hat,p):
    #  based on:
    #  https://lars76.github.io/neural-networks/object-detection/losses-for-segmentation/
    smooth = 1e-3
    gamma = 0 #1.0  0->CE loss
    alpha = 0.5  # 0.25
    fg_loss = (((1-p_hat)**gamma)*p*torch.log(p_hat+smooth)).mean()  #focus on fg
    bg_loss = ((p_hat**gamma)*(1-p)*torch.log(1-p_hat+smooth)).mean()  #focus on bg
    loss = -alpha*fg_loss-(1-alpha)*bg_loss
    #print(bg_loss.item()/(fg_loss.item()+1e-15))
    return loss#,fg_loss.item()/bg_loss.item()

def focal_dice_loss(pred,mask):
    return dice_loss(pred,mask) + binary_focal_loss(pred,mask)


def Lovasz_pytorchBCE(pred,mask):
    from lovasz_loss import LovaszSoftmax
    lovasz = LovaszSoftmax()
    LOVASZ_loss = lovasz(pred,mask)
    pytorchBCE = torch.nn.BCEWithLogitsLoss()
    BCE_loss = pytorchBCE(pred,mask)
    return 0.1*LOVASZ_loss + 0.9*BCE_loss

#TODO
def focal_OHEM(output, target, alpha, gamma, OHEM_percent):
    output = output.contiguous().view(-1)
    target = target.contiguous().view(-1)

    max_val = (-output).clamp(min=0)
    loss = output - output * target + max_val + ((-max_val).exp() + (-output - max_val).exp()).log()

    # This formula gives us the log sigmoid of 1-p if y is 0 and of p if y is 1
    invprobs = F.logsigmoid(-output * (target * 2 - 1))
    focal_loss = alpha * (invprobs * gamma).exp() * loss

    # Online Hard Example Mining: top x% losses (pixel-wise).
    OHEM, _ = focal_loss.topk(k=int(OHEM_percent * [*focal_loss.shape][0]))
    return OHEM.mean()


def dice_BCE(pred,mask):
    BCE = torch.nn.BCEWithLogitsLoss()
    bce_loss = BCE(pred,mask)
    dice_multicls = MulticlassDiceLoss()
    dice = dice_multicls(F.sigmoid(pred),mask)
    return 0.9*bce_loss + 0.1*dice



class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        N = target.size(0)
        smooth = 1

        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N

        return loss


class MulticlassDiceLoss(nn.Module):
    """
    requires one hot encoded target. Applies DiceLoss on each class iteratively.
    requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
      batch size and C is number of classes
    """

    def __init__(self):
        super(MulticlassDiceLoss, self).__init__()

    def forward(self, input, target, weights=None):

        C = target.shape[1]

        # if weights is None:
        # 	weights = torch.ones(C) #uniform weights for all classes

        dice = DiceLoss()
        totalLoss = 0

        for i in range(C):
            diceLoss = dice(input[:, i], target[:, i])
            if weights is not None:
                diceLoss *= weights[i]
            totalLoss += diceLoss
        totalLoss /= C
        return totalLoss
