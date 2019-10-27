import os
import cv2
import pdb
import time
import warnings
import random
import numpy as np

#from model.densenet import densenet121
from model.severstal_cls import Net
from criterion import *
#from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from data import provider
from RAdam import RAdam
#from apex import amp

class Trainer(object):
    '''This class takes care of training and validation of our model'''

    def __init__(self, model):
        self.num_workers = 6
        self.batch_size = {"train": 8, "val": 1}
        self.accumulation_steps = 32 // self.batch_size['train']
        self.base_threshold = 0.5  # <<<<<<<<<<< here's the threshold
        self.net = model

        '''##########hyper-paramaters setting#############'''
        self.lr = 5e-4  #default:5e-4
        self.num_epochs = 80
        self.optim = 'RAdam' #'adam'
        self.learn_plan = 'step'  #step5
        self.loss_function = 'pytorchBCE' #'pytorchBCE'  'Lovasz'
        self.title = 'res34cv'
        self.sel_GPU = '1'  #set None to select both GPU
        self.fold = 4
        '''###############################################'''

        # config path of saving pth
        self.path = os.path.join('weights', self.title, 'f' + str(self.fold))
        if not os.path.isdir(self.path):
            os.mkdir(self.path)
            print('make directory done!!')
        self.file_name = os.path.join(self.path, 'logfile.txt')

        self.data_folder = "input/severstal-steel-defect-detection/"
        self.df_path = 'input/severstal-steel-defect-detection/train.csv'

        self.phases = ["train", "val"]
        self.best_val_loss = float('inf')
        self.best_val_macc = -1
        self.losses = {phase: [] for phase in self.phases}
        self.acc_total = {phase: [] for phase in self.phases}

        ###resume
        resume = False  #<<<<<<<<<<<<<<<<< TODO: whether resume
        if resume:
            weights = torch.load('weights/00007500_model.pth',
                                 map_location=lambda storage, loc: storage)
            self.net.load_state_dict(weights,strict=True)  #["state_dict"]
            print('resuming model done!!!')

        if self.optim == 'adam':
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        elif self.optim == 'SGD':
            self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9)
        elif self.optim == 'RAdam':
            self.optimizer = RAdam(self.net.parameters(), lr=self.lr)

        if self.learn_plan == 'step':
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", patience=3, verbose=True)

        if self.loss_function == 'pytorchBCE':
            self.criterion = torch.nn.BCEWithLogitsLoss() #BCELoss()#BCEWithLogitsLoss()
        elif self.loss_function == 'weighted_BCE':
            self.criterion = weighted_BCE


        print('Now using: ' + self.optim + ' ' + self.learn_plan + ' ' + self.loss_function)

        ##multi GPU
        if self.sel_GPU is not None :
            os.environ['CUDA_VISIBLE_DEVICES'] = self.sel_GPU
        else:
            #self.net, self.optimizer = amp.initialize(self.net.cuda(), self.optimizer, opt_level="O1") #<<<<<<<<<
            os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
            self.net = torch.nn.DataParallel(self.net)
        self.net = self.net.cuda()



        cudnn.benchmark = True
        self.dataloaders = {
            phase: provider(
                data_folder=self.data_folder,
                df_path=self.df_path,
                phase=phase,
                fold=self.fold,
                mean=(0.485, 0.456, 0.406),  # (0.39, 0.39, 0.39),
                std=(0.229, 0.224, 0.225),  # (0.17, 0.17, 0.17),
                batch_size=self.batch_size[phase],
                num_workers=self.num_workers,
            )
            for phase in self.phases
        }


    def forward(self, images, targets):
        images = images.cuda()
        label = targets.cuda()
        label.reqires_grad = False
        outputs = self.net(images)
        outputs = outputs.squeeze(-1).squeeze(-1)
        #if not self.loss_function in ['pytorchBCE', 'mix']:
            #outputs = F.sigmoid(outputs)
        loss = self.criterion(outputs, label)
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
        acc = epoch_log(epoch, phase, self.file_name, epoch_loss, meter) #acc:[4]
        self.losses[phase].append(epoch_loss)
        self.acc_total[phase].append(acc)
        torch.cuda.empty_cache()
        return epoch_loss

    def start(self):
        for epoch in range(self.num_epochs):
            self.iterate(epoch, "train")
            state = {
                "epoch": epoch,
                "best_val_loss": self.best_val_loss,
                #"best_val_macc": self.best_val_macc,
                "state_dict": self.net.state_dict(),
                #"optimizer": self.optimizer.state_dict(),
            }
            val_loss = self.iterate(epoch, "val")
            self.scheduler.step(val_loss)
            acc_total = self.acc_total["val"][-1]

            # when find best dice, the val_loss should not be too large (>1.2*best_val_loss)
            if (val_loss < self.best_val_loss):
                print("******** New optimal found, saving state ********")
                state["best_val_loss"] = self.best_val_loss = val_loss
                M, D, H = time.localtime()[1:4]
                torch.save(state, self.path + '/' + self.title + str(M)+str(D)+str(H)+".pth")
            print()

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    seed = 69
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    #model = densenet121(pretrained=True,num_classes=4)
    model = Net(pretrained=True,num_class=4)
    model_trainer = Trainer(model)
    model_trainer.start()

