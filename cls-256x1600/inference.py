import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from data import TestDataset,post_process,mask2rle
from model_unet_res18 import UNet
from densenet import densenet121
import numpy as np
import torch.nn.functional as F


sample_submission_path = 'input/severstal-steel-defect-detection/sample_submission.csv'
test_data_folder = "input/severstal-steel-defect-detection/test_images"

# initialize test dataloader
cls_threshold = 0.5
seg_threshold = 0.5
num_workers = 6
batch_size = 4
print('threshold:', seg_threshold, cls_threshold)
min_size = 3500
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
df = pd.read_csv(sample_submission_path)
testset = DataLoader(
    TestDataset(test_data_folder, df, mean, std),
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True
)

# Initialize mode and load trained weights
seg_path = "weights/model_seg.pth"
cls_path = "weights/model_cls.pth"
device = torch.device("cuda")

#define model
Seg_model = UNet(n_classes=4)
Seg_model = Seg_model.cuda()

Cls_model = densenet121(pretrained=False,num_classes=1).cuda()
Seg_model.eval()
Cls_model.eval()

#load state dict
#TODO: cls state dict
seg_state = torch.load(seg_path, map_location=lambda storage, loc: storage)
seg_state = seg_state["state_dict"]
for k in list(seg_state.keys()):
    if 'module' == k.split('.')[0]:
        seg_state[k[7:]] = seg_state.pop(k)
Seg_model.load_state_dict(seg_state)

cls_state = torch.load(cls_path, map_location=lambda storage, loc: storage)
cls_state = cls_state["state_dict"]
for k in list(cls_state.keys()):
    if 'module' == k.split('.')[0]:
        cls_state[k[7:]] = cls_state.pop(k)
Cls_model.load_state_dict(cls_state)


# start prediction
predictions = []
for i, batch in enumerate(tqdm(testset)):
    fnames, images = batch
    batch_preds = Seg_model(images.to(device))
    cls_preds = Cls_model(images.to(device))
    batch_preds = batch_preds.detach().cpu().numpy()
    cls_preds = cls_preds.detach().cpu().numpy()

    for fname, masks, posneg in zip(fnames, batch_preds,cls_preds): #N
        #masks:[4,256,1600] posneg:[1]
        c,h,w = masks.shape
        max_conf = masks.max(axis=0)
        max_conf_copy = np.zeros_like(masks)
        for i in range(c):
            max_conf_copy[i] = max_conf
        masks = np.where(masks==max_conf_copy,max_conf_copy,0)

        for cls, pred in enumerate(masks):  #C
            if posneg >= cls_threshold:
                pred, _ = post_process(pred, seg_threshold, min_size) #problem
            else:
                pred = np.zeros_like(pred)
            rle = mask2rle(pred)
            name = fname + "_" + str(cls + 1)  # f"_{cls+1}"
            predictions.append([name, rle])

# save predictions to submission.csv
df = pd.DataFrame(predictions, columns=['ImageId_ClassId', 'EncodedPixels'])
df.to_csv("submission.csv", index=False)
