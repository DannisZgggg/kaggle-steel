import os
import cv2
import pdb
import time
import warnings
import random
import numpy as np

from model.severstal_cls import Net
import segmentation_models_pytorch as smp
from model.severstal_cls import Net
from criterion import *
#from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from data import provider, TTA
from inference import post_process
from RAdam import RAdam
from criterion import dice_loss
#from apex import amp

FOLD = 4


def horizon_trans(img):
    return torch.from_numpy(img.numpy()[:, :, :, ::-1].copy())

def vertical_trans(img):
    return torch.from_numpy(img.numpy()[:, :, ::-1, :].copy())

def rotate_trans(img):
    return horizon_trans(vertical_trans(img))

def none_trans(img):
    return img

def probability_mask_to_probability_label(probability):
    batch_size, num_class, H, W = probability.shape
    probability = probability.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 5)
    value, index = probability.max(1)
    probability = value
    return probability

def TTA_seg(image, seg_model):
    # get average of multi version of test image augmentation
    # image: img:[n,3,256,1600] CPU
    # model: segmentation model
    # output: pred_result:[n,4,256,1600] CPU

    n, c, h, w = image.shape

    trans_zoo = [none_trans, horizon_trans, vertical_trans]  # rotate_trans
    seg_total = torch.zeros((n, 4, h, w))
    sharpen_sel = [1, 0.5, 0.5]

    for i, tran in enumerate(trans_zoo):
        # img->norm+trans->predict->pred_mask->re-trans
        img_aug = tran(image).cuda()
        seg_aug = seg_model(img_aug)
        seg_aug = F.sigmoid(seg_aug)  # [n,4,256,256]
        seg_total += tran(seg_aug.cpu()) ** sharpen_sel[i]  # [n,4,256,256]  sharpen<<<<<<<<<<<<<<<<

    seg_result = seg_total / len(trans_zoo)  # [n,4,256,256]
    return seg_result


def TTA_5cls_seg(image, seg_model):
    outputs = seg_model(image.cuda())
    probability = torch.softmax(outputs, 1)

    probability_mask = probability[:, 1:]  # just drop background
    #probability_label = probability_mask_to_probability_label(probability)[:, 1:]

    probability_mask = probability_mask.data.cpu().numpy()
    #probability_label = probability_label.data.cpu().numpy()

    return probability_mask  #, probability_label  # [n,4,h,w]


def TTA_cls(image, cls_model):
    # get average of multi version of test image augmentation
    # image: img:[n,3,256,1600] CPU
    # model: classification model for input[n,3,256,256]<<<<<<<<
    # output: pred_result:[n,4,256,1600] CPU

    n, c, h, w = image.shape

    trans_zoo = [none_trans, horizon_trans, vertical_trans]  # ,rotate_trans
    sharpen_sel = [1, 0.5, 0.5]
    batch_preds = torch.zeros((n, 4, 1, 1))  # class mask,0 or 1

    for i, tran in enumerate(trans_zoo):
        img_aug = tran(image)
        batch_preds += F.sigmoid(cls_model(img_aug.cuda())).detach().cpu() ** sharpen_sel[i]  # [n,4,1,1]
    batch_preds /= len(trans_zoo)

    return batch_preds

def load_weights(model,path):
    state = torch.load(path, map_location=lambda storage, loc: storage)
    state = state["state_dict"]
    for k in list(state.keys()):
        if 'module' == k.split('.')[0]:
            state[k[7:]] = state.pop(k)
    model.load_state_dict(state)


class Trainer(object):
    '''This class takes care of training and validation of our model'''

    def __init__(self, seg_model,cls_model):
        self.num_workers = 6
        self.batch_size = {"val": 1}


        '''##########hyper-paramaters setting#############'''
        self.num_epochs = 1
        self.title = 'local_dice'
        self.sel_GPU = '0'  #set None to select both GPU
        self.fold = FOLD
        self.cls_threshold = [0.5, 0.5, 0.5, 0.5]
        self.min_size = [600, 600, 1000, 2000]
        self.seg_threshold = [0.5, 0.5, 0.5, 0.5]
        '''###############################################'''

        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.accumulation_steps = 32

        #config path of saving pth
        self.path = os.path.join('weights', self.title, 'f'+str(self.fold))
        if not os.path.isdir(self.path):
            os.mkdir(self.path)
            print('make directory done!!')
        self.file_name = os.path.join(self.path,'logfile.txt')

        self.phases = ["val"]
        #self.device = torch.device("cuda:0")
        #torch.set_default_tensor_type("torch.cuda.FloatTensor")
        self.seg_net = seg_model
        self.cls_net = cls_model


        ##multi GPU
        if self.sel_GPU is not None :
            os.environ['CUDA_VISIBLE_DEVICES'] = self.sel_GPU


        print('Now using: ' + self.title)


        cudnn.benchmark = True
        self.dataloaders = {
            phase: provider(
                data_folder="input/severstal-steel-defect-detection/",
                df_path='input/severstal-steel-defect-detection/train.csv',
                phase=phase,
                fold=self.fold,
                mean=(0.485, 0.456, 0.406),  # (0.39, 0.39, 0.39),
                std=(0.229, 0.224, 0.225),  # (0.17, 0.17, 0.17),
                batch_size=self.batch_size[phase],
                num_workers=self.num_workers,
            )
            for phase in self.phases
        }
        self.losses = {phase: [] for phase in self.phases}
        self.iou_scores = {phase: [] for phase in self.phases}
        self.dice_scores = {phase: [] for phase in self.phases}

    def forward(self, images, targets):
        images = images.cuda()
        masks = targets.cuda()
        masks.reqires_grad = False
        outputs = self.seg_net(images)
        loss = self.criterion(outputs, masks)
        return loss, outputs

    def iterate(self, epoch, phase):
        meter = Meter(phase, epoch)
        print("Starting epoch: "+str(epoch)+" | phase: "+str(phase))
        batch_size = self.batch_size[phase]
        self.seg_net.train(phase == "train")
        dataloader = self.dataloaders[phase]
        running_loss = 0.0
        total_batches = len(dataloader)
        tk0 = tqdm(dataloader, total=total_batches)

        cls_threshold = np.array(self.cls_threshold).reshape((-1,4,1,1))
        for itr, batch in enumerate(tk0):  # replace `dataloader` with `tk0` for tqdm
            with torch.no_grad():
                images, targets = batch

                #seg_result = TTA_5cls_seg(images, self.seg_net).numpy()  # [n,4,h,w]
                seg_result = TTA_seg(images, self.seg_net).numpy()  # [n,4,h,w]
                #cls_result = TTA_cls(images, self.cls_net).numpy()
                #cls_result = (cls_result > cls_threshold).astype(np.float32)
                cls_result = np.ones_like(cls_threshold)
                batch_preds = seg_result * cls_result  # [n,4,256,1600]

            for n, masks in enumerate(batch_preds):  # N
                # masks:[C,256,1600]
                for c, pred in enumerate(masks):  # C
                    # #pred, _ = post_process(pred, self.seg_threshold[c], self.min_size[c])
                    pred = (pred > self.seg_threshold[c]).astype(np.uint8)
                    if (pred.sum() < self.min_size[c]):
                        pred = np.zeros(pred.shape, dtype=pred.dtype)
                    batch_preds[n,c] = pred

            batch_preds = torch.from_numpy(batch_preds).float()
            meter.update(targets, batch_preds)
            loss = self.criterion(batch_preds,targets)
            running_loss += loss.item()
            tk0.set_postfix(loss=loss.item())
        epoch_loss = running_loss / total_batches
        dice, iou = epoch_log(epoch, phase, self.file_name, epoch_loss, meter)
        self.losses[phase].append(epoch_loss)
        self.dice_scores[phase].append(dice)
        self.iou_scores[phase].append(iou)
        torch.cuda.empty_cache()
        return epoch_loss

    def start(self):
        for epoch in range(self.num_epochs):
            self.iterate(epoch, "val")



if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    seed = 69
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


    #segmodel = smp.FPN('efficientnet-b5',encoder_weights=None,classes=5).cuda()
    # segmodel = smp.Unet('resnet34',encoder_weights=None,classes=4).cuda()
    # seg_path = '../kaggle-256crop-4cls-seg/weights/resnet34/'
    # fold_mdoel = [
    #     'f0/resnet34_f0_10422.pth',
    #     'f1/resnet34_f1_10512.pth',
    #     'f2/resnet34_f2_1060.pth',
    #     'f3/resnet34_f3_1063.pth',
    #     'f4/resnet34_f4_10611.pth'
    # ]

    segmodel = smp.FPN('efficientnet-b5',encoder_weights=None,classes=4).cuda() #se_resnet50
    seg_path = '../kaggle-256crop-4cls-seg/weights/effb5FPN/'
    fold_mdoel = [
        'f0/effb5FPN_f0_101312.pth',
        'f1/effb5FPN_f1_10148.pth',
        'f2/effb5FPN_f2_10148.pth',
        'f3/effb5FPN_f3_10153.pth',
        'f4/effb5FPN_f4_10164.pth'
    ]

    load_weights(segmodel,seg_path+fold_mdoel[FOLD]) #<<<<<<<<<  set fold
    segmodel.eval()


    clsmodel = Net().cuda()
    #print(model)
    model_trainer = Trainer(segmodel,clsmodel)
    model_trainer.start()
