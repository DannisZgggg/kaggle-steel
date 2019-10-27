# refer to: https://www.kaggle.com/iafoss/256x256-images-with-defects
import gc
import os
import cv2
from tqdm import tqdm
import zipfile
import io
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


sz = (1600, 256)  # size of input images
sz0 = 256
MASKS = 'input/severstal-steel-defect-detection/train.csv'
IMAGES = 'input/severstal-steel-defect-detection/train_images/'


def enc2mask(encs, shape=(1600, 256)):
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for m, enc in enumerate(encs):
        if isinstance(enc, np.float) and np.isnan(enc):
            continue
        s = enc.split()
        for i in range(len(s)//2):
            start = int(s[2*i]) - 1
            length = int(s[2*i+1])
            img[start:start+length] = 1 + m
    return img.reshape(shape).T


def mask2enc(mask, shape=(256, 256), n=4):
    pixels = mask.T.flatten()
    encs = []
    global image_name, mask_enc
    for i in range(1, n+1):
        p = (pixels == i).astype(np.int8)
        if p.sum() == 0:
            # image_name.append(name+'_'+str(i))
            encs.append(np.nan)
        else:
            p = np.concatenate([[0], p, [0]])
            runs = np.where(p[1:] != p[:-1])[0] + 1
            runs[1::2] -= runs[::2]
            encs.append(' '.join(str(x) for x in runs))
    return encs


df_masks = pd.read_csv(MASKS)
df_masks['id'] = [id[:-2] for id in df_masks.ImageId_ClassId]
df_masks = pd.DataFrame(df_masks.groupby('id')['EncodedPixels'].apply(list))
df_masks.head()
fnames = os.listdir(IMAGES)
save_image = 'severstal-256-crop/crop_images/'


# split each image in 6 parts with cutting 32 pixels from left and right
n_crops = 6
offsets = [32 + sz0 * i for i in range(n_crops)]

image_name = []
mask_enc = []
x_tot, x2_tot = [], []
for fname in tqdm(fnames):
    img0 = Image.open(os.path.join(IMAGES, fname))
    img0 = np.asarray(img0)
    mask0 = enc2mask(df_masks.loc[fname].EncodedPixels)
    for i in range(n_crops):
        img = img0[:, offsets[i]:offsets[i] + sz0, :1]  # keep only one channel
        mask = mask0[:, offsets[i]:offsets[i] + sz0]
        # if mask.max() == 0 and fname not in TEXTURE_IMGS:
        if mask.max() == 0:
            name = fname[:-4] + '_' + str(i) + '.jpg'

            #encs = mask2enc(mask)
            # for i in range(4):
            #image_name.append(name + '_' + str(i + 1))
            # mask_enc.append(encs[i])
            cv2.imwrite(save_image + name, img)

        else:
            name = fname[:-4] + '_' + str(i) + '.jpg'
            x_tot.append((img / 255.0).mean())
            x2_tot.append(((img / 255.0) ** 2).mean())
            encs = mask2enc(mask)
            for i in range(4):
                image_name.append(name + '_' + str(i + 1))
                mask_enc.append(encs[i])
            cv2.imwrite(save_image + name, img)


# positive
test_dict = {'ImageId_ClassId': image_name, 'EncodedPixels': mask_enc}
test_dict_df = pd.DataFrame(test_dict)


# hard negative
pred = pd.read_csv('pred.csv')
pred_fname = list(pred.head(12000).fname)

hard_neg = []
hard_neg_value = []
for f in tqdm(pred_fname):
    hard_neg.append(f.split('.')[0]+'.jpg'+'_1')
    hard_neg.append(f.split('.')[0]+'.jpg'+'_2')
    hard_neg.append(f.split('.')[0]+'.jpg'+'_3')
    hard_neg.append(f.split('.')[0]+'.jpg'+'_4')
    hard_neg_value.append(np.nan)
    hard_neg_value.append(np.nan)
    hard_neg_value.append(np.nan)
    hard_neg_value.append(np.nan)

test_dict_neg = {'ImageId_ClassId': hard_neg, 'EncodedPixels': hard_neg_value}
test_dict_df_neg = pd.DataFrame(test_dict_neg)


# merge two dataframe
df_new = pd.concat([test_dict_df, test_dict_df_neg], ignore_index=True)
df_new.to_csv('severstal-256-crop/crop_256.csv', index=0)
