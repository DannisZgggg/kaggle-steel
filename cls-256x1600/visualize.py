import pandas as pd
import cv2
from data import make_mask
from albumentations import *
import numpy as np
import matplotlib.pyplot as plt

def load_data():
    df = pd.read_csv("input/severstal-steel-defect-detection/train.csv")

    df['ImageId'], df['ClassId'] = zip(*df['ImageId_ClassId'].str.split('_'))
    df['ClassId'] = df['ClassId'].astype(int)
    df = df.pivot(index='ImageId', columns='ClassId', values='EncodedPixels')
    df['defects'] = df.count(axis=1)
    return df

if __name__ == '__main__':
    img_name, mask = make_mask(59, load_data())
    img_ori = cv2.imread("input/severstal-steel-defect-detection/train_images/" + img_name)
    img_save = img_ori.copy()

    '''kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    for ch in range(4):
        mask[:, :, ch] = cv2.morphologyEx(mask[:, :, ch], cv2.MORPH_OPEN, kernel)'''


    list_trfms = Compose([ShiftScaleRotate(p=1,shift_limit=0.1, scale_limit=0.5, rotate_limit=0)])
    res = list_trfms(image = img_ori, mask = mask)
    img_tran = res['image']
    mask_tran = res['mask']

    img_ori = img_ori[:,:,0]
    img_tran = img_tran[:,:,0]

    #仅针对第0类
    ch = 0  # <<<<<<<define class

    #如果可以转换
    if np.sum(mask_tran[:,:,ch])>0:
        ori_mean = np.mean(img_ori*mask[:,:,ch]) #原来的部分的均值
        tran_mean = np.mean(img_ori*mask_tran[:,:,ch]) #转变后的部分的均值
        sub = tran_mean - ori_mean
        print(sub)
        img_tran = img_tran + sub



    if np.sum(img_ori[(mask_tran>0)[:,:,ch]]<10)<100: #黑色像素点少则可以贴图
        print('copy')
    else:
        print('not copy')
    mask+=mask_tran
    img_ori[(mask_tran>0)[:,:,ch]] = img_tran[(mask_tran>0)[:,:,ch]]

    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(mask_tran, kernel, iterations=1)
    dilation = cv2.dilate(mask_tran, kernel, iterations=1)




    cv2.imshow('img', img_save)
    cv2.imshow('img_tran', img_ori)
    cv2.waitKey(0)

