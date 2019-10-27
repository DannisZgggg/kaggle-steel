import torch
import pandas as pd
from tqdm import tqdm
import cv2
import numpy as np
from torch.utils.data import DataLoader, Dataset
from data import TestDataset,mask2rle
from albumentations import Compose,Normalize,HorizontalFlip,VerticalFlip
from model_kaggle import Unet
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

def TTA(image,model):
    #get average of multi version of test image augmentation
    #batch size must be 1
    #imput: img:[256,1600,3],torch.Tensor
    #output: pred_result:[4,256,1600],np.array
    h,w,c = image.shape
    horizon_trans = Compose([HorizontalFlip(p=1)])
    vertical_trans = Compose([VerticalFlip(p=1)])
    rotate_trans = Compose([HorizontalFlip(p=1),VerticalFlip(p=1)])
    none_trans = Compose([])
    trans_zoo = [horizon_trans,vertical_trans,rotate_trans,none_trans]
    pred_total = np.empty((len(trans_zoo),h,w,4))

    for i,tran in enumerate(trans_zoo):
        #img->norm+trans->predict->pred_mask->re-trans
        #numpy.array
        img_aug = tran(image=image.numpy())['image'].squeeze() #[256,1600,3]
        #img_aug = normal_trans(image=img_aug)['image'].squeeze()
        img_aug = torch.from_numpy(img_aug).permute((2,0,1)).unsqueeze(0).cuda() #[1,3,256,1600]
        pred_aug = model(img_aug)
        pred_aug = F.sigmoid(pred_aug).detach().cpu().numpy()#[1,4,256,1600]
        pred_aug = pred_aug.squeeze().transpose((1,2,0)) #[256,1600,4]
        pred_recover = tran(image=pred_aug)['image'].squeeze() #[256,1600,4]
        pred_total[i] = pred_recover
    pred_result = np.mean(pred_total,axis=0) #[256,1600,4]
    return pred_result.transpose((2,0,1)) #[4,256,1600]


def post_process(probability, threshold, min_size):
    '''Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored'''
    mask = (probability>threshold)
    #cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((256, 1600), np.float32)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num


if __name__ == '__main__':
    sample_submission_path = 'input/severstal-steel-defect-detection/sample_submission.csv'
    test_data_folder = "input/severstal-steel-defect-detection/test_images"

    # initialize test dataloader
    best_threshold = [0.5,0.5,0.55,0.55]
    num_workers = 6
    batch_size = 4
    print('best_threshold', best_threshold)
    min_size = [800,2200,1000,3800]
    mean = (0.485, 0.456, 0.406),  # (0.39, 0.39, 0.39),
    std = (0.229, 0.224, 0.225),  # (0.17, 0.17, 0.17),
    df = pd.read_csv(sample_submission_path)
    testset = DataLoader(
        TestDataset(test_data_folder, df, mean, std),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # Initialize mode and load trained weights
    ckpt_path = "weights/model_RAdamsteppytorchBCE970.pth"
    device = torch.device("cuda")
    model = Unet("resnet18", encoder_weights=None, classes=4, activation=None).to(device)

    model.eval()
    state = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    state = state["state_dict"]
    for k in list(state.keys()):
        if 'module' == k.split('.')[0]:
            state[k[7:]] = state.pop(k)
    model.load_state_dict(state)

    # start prediction
    use_TTA = True
    predictions = []
    for i, batch in enumerate(tqdm(testset)):
        fnames, images = batch
        n, h, w, c = images.shape #[n,h,w,3]
        if not use_TTA:
            batch_preds = model(images.permute((0,3,1,2)).to(device))
            batch_preds = F.sigmoid(batch_preds).detach().cpu().numpy() #[n,c,h,w]
        else:
            batch_preds = np.empty((n,4,h,w))
            for b in range(images.shape[0]):
                batch_preds[b] = TTA(images[b],model)

        for fname, preds in zip(fnames, batch_preds): #preds:[c,h,w]
            for cls, pred in enumerate(preds):
                pred, _ = post_process(pred, best_threshold[cls], min_size[cls]) #pred:[h,w]
                rle = mask2rle(pred)
                name = fname + "_" + str(cls+1)#f"_{cls+1}"
                predictions.append([name, rle])

    # save predictions to submission.csv
    df = pd.DataFrame(predictions, columns=['ImageId_ClassId', 'EncodedPixels'])
    df.to_csv("submission.csv", index=False)
