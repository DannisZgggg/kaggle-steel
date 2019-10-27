import os
import cv2
import pdb
import time
import warnings
import random
import numpy as np

import segmentation_models_pytorch as smp
from model.severstal_cls import Net
from criterion import *
#from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from data import provider
from inference import post_process

FOLD = 1

warnings.filterwarnings("ignore")
seed = 69
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

def rid_bg_layer(mask):
    return mask[:,1:,:,:]


def TTA_seg(image, seg_model):
    # get average of multi version of test image augmentation
    # image: img:[n,3,256,1600] CPU
    # model: segmentation model
    # output: pred_result:[n,4,256,1600] CPU

    def horizon_trans(img):
        return torch.from_numpy(img.numpy()[:, :, :, ::-1].copy())

    def vertical_trans(img):
        return torch.from_numpy(img.numpy()[:, :, ::-1, :].copy())

    def rotate_trans(img):
        return horizon_trans(vertical_trans(img))

    def none_trans(img):
        return img

    n, c, h, w = image.shape

    trans_zoo = [horizon_trans, none_trans]  # ,vertical_trans,rotate_trans
    seg_total = torch.zeros((n, 4, h, w))

    for tran in trans_zoo:
        # img->norm+trans->predict->pred_mask->re-trans
        img_aug = tran(image).cuda()
        seg_aug = seg_model(img_aug)
        seg_aug = F.sigmoid(seg_aug)  # [n,4,256,256]
        seg_total += tran(seg_aug.cpu())  # [n,4,256,256]  sharpen<<<<<<<<<<<<<<<<

    seg_result = seg_total / len(trans_zoo)  # [n,4,256,256]
    return seg_result


def TTA_5cls_seg(image, seg_model):
    outputs = seg_model(image.cuda())
    probability = torch.softmax(outputs, 1)

    probability_mask = probability[:, 1:]  # just drop background
    #probability_label = probability_mask_to_probability_label(probability)[:, 1:]

    probability_mask = probability_mask.data.cpu()
    #probability_label = probability_label.data.cpu().numpy()

    return probability_mask  #, probability_label  # [n,4,h,w]

def TTA_cls(image, cls_model):
    # get average of multi version of test image augmentation
    # image: img:[n,3,256,1600] CPU
    # model: classification model for input[n,3,256,256]<<<<<<<<
    # output: pred_result:[n,4,256,1600] CPU

    def horizon_trans(img):
        return torch.from_numpy(img.numpy()[:, :, :, ::-1].copy())

    def vertical_trans(img):
        return torch.from_numpy(img.numpy()[:, :, ::-1, :].copy())

    def rotate_trans(img):
        return horizon_trans(vertical_trans(img))

    def none_trans(img):
        return img

    n, c, h, w = image.shape

    trans_zoo = [horizon_trans, none_trans]  # ,vertical_trans,rotate_trans
    batch_preds = torch.zeros((n, 4, h, w))  # class mask,0 or 1
    msize = 64
    den = torch.ones_like(batch_preds) * 5  # [n,4,256,1600]
    ones_block = torch.ones((n, 4, 256, 256))
    for m in range(4, 0, -1):  # 4,3,2,1
        den[:, :, :, 0:m * msize] = m
        den[:, :, :, -m * msize:] = m
    # 1600//64=25  25-3=22
    disp = 0
    for move in range(22):
        images_block = image[:, :, :, disp:disp + 256]
        posneg = torch.zeros((n, 4))
        for tran in trans_zoo:
            img_aug = tran(images_block)
            posneg += F.sigmoid(cls_model(img_aug.cuda())).detach().cpu()  # [n,4]
        posneg /= len(trans_zoo)
        posneg = posneg.unsqueeze(-1).unsqueeze(-1).float()  # [n,4,1,1]
        batch_preds[:, :, :, disp:disp + 256] += (ones_block * posneg)
        disp += msize
    batch_preds /= den
    return batch_preds


class Trainer(object):
    '''This class takes care of training and validation of our model'''

    def __init__(self, seg_model, cls_model):
        self.num_workers = 6
        self.batch_size = {"val": 4}

        '''##########hyper-paramaters setting#############'''
        self.num_epochs = 1
        self.title = 'grid_search'
        self.sel_GPU = '0'  # set None to select both GPU
        self.fold = FOLD
        self.cls_thresholds = -1
        self.min_sizes = np.linspace(0, 3200, 17)
        self.seg_thresholds = np.array([0.25, 0.40, 0.50, 0.65, 0.70, 0.80, 0.85, 0.90, 0.95])
        '''###############################################'''

        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.accumulation_steps = 32

        # config path of saving pth
        self.path = os.path.join('weights', self.title, 'f' + str(self.fold))
        self.file_name = os.path.join(self.path, 'logfile.txt')

        self.phases = ["val"]
        # self.device = torch.device("cuda:0")
        # torch.set_default_tensor_type("torch.cuda.FloatTensor")

        self.seg_net = seg_model
        self.cls_net = cls_model

        ##multi GPU
        if self.sel_GPU is not None:
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
        print("Starting epoch: " + str(epoch) + " | phase: " + str(phase))
        batch_size = self.batch_size[phase]
        dataloader = self.dataloaders[phase]
        running_loss = 0.0
        total_batches = len(dataloader)
        tk0 = tqdm(dataloader, total=total_batches)
        n_seg_thre = self.seg_thresholds.shape[0]
        n_min_size = self.min_sizes.shape[0]

        dice_grid = np.zeros((4, n_seg_thre, n_min_size))

        def dice_layer(pred, mask):
            # both are one-hot
            sigma = 1
            return (2 * (pred * mask).sum() + sigma) / (pred.sum() + mask.sum() + sigma)

        for itr, batch in enumerate(tk0):  # replace `dataloader` with `tk0` for tqdm
            images, targets = batch
            targets = targets.numpy()
            bs = images.shape[0]

            #seg_result = TTA_seg(images, self.seg_net).numpy()  # [n,4,256,1600]
            seg_result = TTA_5cls_seg(images, self.seg_net).numpy()  # [n,4,256,1600]
            cls_result = np.zeros((bs, 4))  # TTA_cls(images, self.cls_net).numpy()  # [n,4,1,1]

            for n, (seg_n, cls_n) in enumerate(zip(seg_result, cls_result)):  # N   [C,256,1600], C
                for c, (seg_c, cls_c) in enumerate(zip(seg_n, cls_n)):  # C  [256,1600], 1
                    for st, seg_thre in enumerate(self.seg_thresholds):
                        for ms, min_size in enumerate(self.min_sizes):
                            pred_thre = (seg_c > seg_thre).astype(np.uint8)
                            if pred_thre.sum() < min_size:
                                pred_thre = np.zeros(pred_thre.shape, dtype=pred_thre.dtype)
                            # pred_thre = seg_c   #(seg_c * (cls_c>cls_thre)).astype(np.float32)
                            # pred_post, _ = post_process(pred_thre, seg_thre, min_size)
                            dice_grid[c, st, ms] += dice_layer(pred_thre, targets[n, c])
            #if itr==10: break
        dice_grid /= ((itr + 1) * self.batch_size["val"])
        torch.cuda.empty_cache()
        return dice_grid

    def start(self):
        with torch.no_grad():
            return self.iterate(0, "val")


class Model:
    def __init__(self, models):
        self.models = models

    def __call__(self, x):
        res = []
        x = x.cuda()
        with torch.no_grad():
            for m in self.models:
                res.append(m(x))
        res = torch.stack(res)
        return torch.mean(res, dim=0)

def load_weights(model,path):
    state = torch.load(path, map_location=lambda storage, loc: storage)
    state = state["state_dict"]
    for k in list(state.keys()):
        if 'module' == k.split('.')[0]:
            state[k[7:]] = state.pop(k)
    model.load_state_dict(state)


cls_path = '../kaggle-256crop-4cls-seg/weights/'

# Initialize mode and load trained weights
Cls_model = Net(pretrained=False).cuda()
Cls_model.eval()
load_weights(Cls_model,os.path.join(cls_path,"resnet34_pytorchBCERAdamsteppytorchBCE92914.pth"))

############################################
# Seg_model = smp.FPN('efficientnet-b5',encoder_weights=None,classes=5).cuda()
#
# weights_name = ['f0/Heng-efficient_f0_1076.pth',
#                 'f1/Heng-efficient_f1_10719.pth',
#                 'f2/Heng-efficient_f2_1085.pth',
#                 'f3/Heng-efficient_f3_1086.pth',
#                 'f4/Heng-efficient_f4_10818.pth',]
# seg_path = '../kaggle-steel-0911/weights/'
#
# load_weights(Seg_model,os.path.join(seg_path,'Heng-efficient',weights_name[FOLD]))#resnet34_f4_10611

# Seg_model = Model([seg4])
# for net in Seg_model.models:
#     net.eval()

Seg_model = smp.FPN('resnet34',encoder_weights=None,classes=4,activation='sigmoid').cuda()
load_weights(Seg_model, "../kaggle-256crop-4cls-seg/weights/res34FPN/f0/res34FPN_f0_10126.pth")

print('load weights done!!')


model_trainer = Trainer(Seg_model, Cls_model)
dice_grid = model_trainer.start()
print('grid search done!!')
np.save('grid_search.npy',dice_grid)



