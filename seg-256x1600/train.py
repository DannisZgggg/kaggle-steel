import os
import cv2
import pdb
import time
import warnings
import random
import numpy as np
#from model.model_unet_res18 import UNet
#from model_resnext50 import UNet
#from model.model_kaggle import Unet
#from model.TGS_model import SteelUnetv5
#from model.TGS_v2 import Unet_scSE_hyper
from model.seresnet50 import Unet
import segmentation_models_pytorch as smp
from criterion import *
#from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from data import provider
from SGDR import CosineAnnealingWithRestartsLR
from RAdam import RAdam
from criterion import dice_loss
#from apex import amp

class Trainer(object):
    '''This class takes care of training and validation of our model'''

    def __init__(self, model):
        self.num_workers = 6
        self.batch_size = {"train": 8, "val": 2}
        self.accumulation_steps = 32 // self.batch_size['train']
        self.base_threshold = 0.5  # <<<<<<<<<<< here's the threshold

        '''##########hyper-paramaters setting#############'''
        self.lr = 5e-4  #default:5e-4
        self.num_epochs = 80
        self.optim = 'RAdam' #'adam'
        self.learn_plan = 'step'  #step5
        self.loss_function = 'pytorchBCE' #'pytorchBCE'  'Lovasz'
        self.title = 'seresnet50'
        self.crop256 = True
        '''###############################################'''


        #config path of saving pth
        path = os.path.join('weights', self.title)
        if not os.path.isdir(path):
            os.mkdir(path)
            print('make directory done!!')
        self.file_name = os.path.join(path,'logfile.txt')


        #config whether to crop
        if self.crop256:
            self.data_folder = "severstal-256-crop/"
            self.df_path = 'severstal-256-crop/crop_256.csv'
        else:
            self.data_folder = "input/severstal-steel-defect-detection/"
            self.df_path = 'input/severstal-steel-defect-detection/train.csv'


        self.best_loss = float("inf")
        self.phases = ["train", "val"]
        #self.device = torch.device("cuda:0")
        #torch.set_default_tensor_type("torch.cuda.FloatTensor")
        self.net = model

        ###resume
        resume = False  #<<<<<<<<<<<<<<<<< TODO: whether resume
        if resume:
            weights = torch.load('weights/model_RAdamsteppytorchBCE9150.pth',
                                 map_location=lambda storage, loc: storage)
            self.net.load_state_dict(weights["state_dict"],strict=True)
            print('resuming model done!!!')

        if self.optim == 'adam':
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        elif self.optim == 'SGD':
            self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9)
        elif self.optim == 'RAdam':
            self.optimizer = RAdam(self.net.parameters(), lr=self.lr)

        if self.learn_plan == 'step':
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", patience=3, verbose=True)
        elif self.learn_plan == 'SGDR':
            self.scheduler = CosineAnnealingWithRestartsLR(self.optimizer,T_max=5)  #set lr=1e-3
            #self.lr = 1e-3

        if self.loss_function == 'pytorchBCE':
            self.criterion = torch.nn.BCEWithLogitsLoss() #BCELoss()#BCEWithLogitsLoss()
        elif self.loss_function == 'dice_loss':
            self.criterion = dice_loss
        elif self.loss_function == 'weighted_BCE_loss':
            self.criterion = weighted_BCE_loss
        elif self.loss_function == 'binary_focal_loss':
            self.criterion = binary_focal_loss
        elif self.loss_function == 'generalized_dice_loss':
            self.criterion = generalized_dice_loss
        elif self.loss_function == 'Lovasz':
            from  lovasz_loss import LovaszSoftmax
            self.criterion = LovaszSoftmax()
        elif self.loss_function == 'mix':
            self.criterion = dice_BCE
        elif self.loss_function == 'change':
            pass



        ##multi GPU
        self.mGPU_apex = True  #<<<<<<<<<<<<<<<<<<<TODO: set mGPU here
        if self.mGPU_apex:
            #self.net, self.optimizer = amp.initialize(self.net.cuda(), self.optimizer, opt_level="O1") #<<<<<<<<<
            self.net = torch.nn.DataParallel(self.net)
        self.net = self.net.cuda()


        print('Now using: ' + self.optim + ' ' + self.learn_plan + ' ' + self.loss_function)
        self.best_val_loss = float('inf')
        self.best_val_dice = -1


        cudnn.benchmark = True
        self.dataloaders = {
            phase: provider(
                data_folder=self.data_folder,
                df_path=self.df_path,
                phase=phase,
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
        outputs = self.net(images)
        #if not self.loss_function in ['pytorchBCE', 'mix']:
            #outputs = F.sigmoid(outputs)
        loss = self.criterion(outputs, masks)
        return loss, outputs

    def iterate(self, epoch, phase):
        meter = Meter(phase, epoch)
        print("Starting epoch: "+str(epoch)+" | phase: "+str(phase))
        batch_size = self.batch_size[phase]
        self.net.train(phase == "train")
        dataloader = self.dataloaders[phase]
        running_loss = 0.0
        total_batches = len(dataloader)
        tk0 = tqdm(dataloader, total=total_batches)

        self.optimizer.zero_grad()
        for itr, batch in enumerate(tk0):  # replace `dataloader` with `tk0` for tqdm
            images, targets = batch
            '''targets_smooth = torch.where(targets>0.5,
                 0.85*torch.ones_like(targets),
                 0.05*torch.ones_like(targets)) ##TODO'''
            loss, outputs = self.forward(images, targets)
            #if self.loss_function in ['pytorchBCE', 'mix']:
            outputs = F.sigmoid(outputs)
            loss = loss / self.accumulation_steps
            if phase == "train":
                loss.backward()
                #with amp.scale_loss(loss, self.optimizer) as scaled_loss: #<<<<<<<<<<
                    #scaled_loss.backward()
                if (itr + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            running_loss += loss.item()
            outputs = outputs.detach().cpu()
            meter.update(targets, outputs)
            tk0.set_postfix(loss=loss.item())
        epoch_loss = (running_loss * self.accumulation_steps) / total_batches
        dice, iou = epoch_log(epoch, phase, self.file_name, epoch_loss, meter)
        self.losses[phase].append(epoch_loss)
        self.dice_scores[phase].append(dice)
        self.iou_scores[phase].append(iou)
        torch.cuda.empty_cache()
        return epoch_loss

    def start(self):
        for epoch in range(self.num_epochs):
            self.iterate(epoch, "train")
            state = {
                "epoch": epoch,
                "best_val_loss": self.best_val_loss,
                "best_val_dice": self.best_val_dice,
                "state_dict": self.net.state_dict(),
                #"optimizer": self.optimizer.state_dict(),
            }
            val_loss = self.iterate(epoch, "val")
            self.scheduler.step(val_loss)
            val_dice = self.dice_scores["val"][-1]
            # when find best dice, the val_loss should not be too large (>1.2*best_val_loss)
            if (val_dice >= self.best_val_dice and val_loss < 1.2*self.best_val_loss) \
                or (val_loss < self.best_val_loss):
                print("******** New optimal found, saving state ********")
                state["best_val_loss"] = self.best_val_loss = min(val_loss,self.best_val_loss)
                self.best_val_dice = max(val_dice, self.best_val_dice)
                M, D, H = time.localtime()[1:4]
                torch.save(state, "weights/model_"+self.optim+self.learn_plan+self.loss_function+
                           str(M)+str(D)+str(H)+".pth")
            print()

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    seed = 69
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    #model = UNet(n_classes=4,phase='train')
    #model = Unet("resnet18", encoder_weights="imagenet", classes=4, activation=None)
    #model = SteelUnetv5(n_classes=4)
    #model = Unet_scSE_hyper(n_classes=4,pretrained=True)
    model = smp.Unet('se_resnet50',encoder_weights="imagenet",classes=4,activation='sigmoid')
    #print(model)
    model_trainer = Trainer(model)
    model_trainer.start()
