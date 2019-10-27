import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable


def predict(X, threshold):
    '''X is sigmoid output of the model'''
    X_p = np.copy(X)
    threshold = np.array(threshold).reshape(-1,4)
    preds = (X_p > threshold).astype(np.float32)
    return preds


class Meter:
    '''A meter to keep track of iou and dice scores throughout an epoch'''
    # accuracy for all labels including fg, bg
    # def __init__(self, phase, epoch):
    #     self.base_threshold = [0.5, 0.5, 0.5, 0.5] # <<<<<<<<<<< here's the threshold
    #     self.TP = np.zeros((1,4)) #[num_batch,num_classes]


    # def update(self, targets, outputs):
    #     targets = targets.numpy().astype(np.float32)
    #     outputs = predict(outputs,self.base_threshold)
    #     equal = (targets==outputs).astype(np.float32) #[n,c]
    #     self.TP = np.concatenate((self.TP,equal),axis=0)
    #
    # def get_metrics(self):
    #     accuracy = np.mean(self.TP,axis=0)
    #     return accuracy

    #accuracy just for fg
    def __init__(self, phase, epoch):
        self.base_threshold = [0.5, 0.5, 0.5, 0.5] # <<<<<<<<<<< here's the threshold
        self.pred = []
        self.targ = []


    def update(self, targets, outputs):  #[n,c]
        targets = targets.numpy().astype(np.float32)
        outputs = predict(outputs,self.base_threshold)
        self.pred.append(outputs)
        self.targ.append(targets)


    def get_metrics(self):
        self.pred = np.concatenate(self.pred, axis=0)
        self.targ = np.concatenate(self.targ, axis=0)
        tp = self.pred * self.targ
        tp = np.sum(tp,axis=0) #[4]

        tpfp = np.sum(self.pred,axis=0)
        tpfn = np.sum(self.targ,axis=0)

        precision = tp/tpfp
        # recall = tp/tpfn
        #
        # n_equal = np.sum(self.pred!=self.targ,axis=0)
        # print('num_data:',self.pred.shape[0])
        # print('num_target:',np.sum(self.targ,axis=0))
        # print('                     TP:', tp)
        # print('precision = TP/predictP:', precision)
        # print('recall    = TP/targetP :', recall)
        # print('not equal              :', n_equal)

        return precision

def epoch_log(epoch, phase, file_name, epoch_loss, meter):
    '''logging the metrics at the end of an epoch'''
    acc = meter.get_metrics()

    info = "Loss: %0.4f | Accuracy: %0.4f, %0.4f, %0.4f, %0.4f" % \
           (epoch_loss, acc[0],acc[1],acc[2],acc[3])
    print(info)
    f = open(file_name, 'a')
    if phase == 'train':
        f.write('epoch:' + str(epoch) + '\n')
    f.write(info + '\n')
    if phase == 'val':
        f.write('\n')
    f.close()
    #print(info)   <<<<<<<<<<

    return acc


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


def weighted_BCE(logit, truth, weight=[0.75,0.25]): #[0.75,0.25]

    loss = F.binary_cross_entropy_with_logits(logit, truth, reduction='none')

    if weight is None:
        loss = loss.mean()

    else:
        pos = (truth>0.5).float()
        neg = (truth<0.5).float()
        pos_sum = pos.sum().item() + 1e-12
        neg_sum = neg.sum().item() + 1e-12
        loss = (weight[1]*pos*loss/pos_sum + weight[0]*neg*loss/neg_sum).mean()
        #raise NotImplementedError

    return loss

