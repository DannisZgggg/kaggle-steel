import os
import cv2
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, RandomSampler, SequentialSampler
import torch
from albumentations import *
from albumentations.torch import ToTensor
from albumentations.augmentations.functional import rotate
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import jpeg4py as jpeg


#https://www.kaggle.com/paulorzp/rle-functions-run-lenght-encode-decode
def mask2rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def make_mask(row_id, df):
    '''Given a row index, return image_id and mask (256, 256, 4)'''
    fname = df.iloc[row_id].name
    labels = df.iloc[row_id][:4]
    masks = np.zeros((256, 256, 4), dtype=np.float32) # float32 is V.Imp
    # 4:class 1～4 (ch:0～3)

    for idx, label in enumerate(labels.values):
        if label is not np.nan:
            label = label.split(" ")
            positions = map(int, label[0::2])
            length = map(int, label[1::2])
            mask = np.zeros(256 * 256, dtype=np.uint8)
            for pos, le in zip(positions, length):
                mask[pos:(pos + le)] = 1
            masks[:, :, idx] = mask.reshape(256, 256, order='F')
    return fname, masks


def img_possess(img):
    #np.array [4,256,256]
    kernel = np.array(
        [[0, -1, 0],
         [-1, 4, -1],
         [0, -1, 0]]
    )
    kernel2 = np.array(
        [[-1, 0, 1],
         [-1, 0, 1],
         [-1, 0, 1]]
    )
    # plan1:DOG
    blur = cv2.GaussianBlur(img[:, :, 0], (3, 3), 0)
    img[:, :, 1] = cv2.filter2D(blur, -1, kernel)
    # plan2:Prewitt Horizontal
    img[:, :, 2] = cv2.filter2D(img[:, :, 0], -1, kernel2)
    return img



def aug_patch_smooth(img,mask):
    '''
    :param img: [256,256]
    :param mask: [256,256,4]
    :return: augmented img and mask, same shape as ori
    '''
    list_trfms = Compose([ShiftScaleRotate(p=1, shift_limit=0.3, scale_limit=0.3, rotate_limit=10)])
    res = list_trfms(image=img, mask=mask)
    img_tran = res['image']  # [256,256]
    mask_tran = res['mask']  # [256,256,4]

    #####仅对第2类进行增强######(从0数起)
    if (mask_tran.sum(0).sum(0)[2] > 0):  # 如果tran后第2类有mask
        for ch in [2]:  # <<<<<<<<<<<<定义类的种类
            num_component, component = cv2.connectedComponents(mask_tran[:, :, ch].astype(np.uint8))
            for c in range(1, num_component):  # 各个变换后的patch减去均值之差
                p = (component == c).astype('uint8')
                back_mean = img[p == 1].sum() / (p == 1).sum()
                if back_mean < 10:
                    mask_tran[:, :, ch][p == 1] = 0
                else:
                    kernel = np.ones((3, 3), np.uint8)
                    # erosion = cv2.erode(mask_tran[:, :, ch], kernel, iterations=1)
                    dilation = cv2.dilate(p, kernel, iterations=1)
                    loop = dilation - p
                    loop_mean = img_tran[loop == 1].sum() / loop.sum()
                    img_tran[p == 1] = np.clip((img_tran[p == 1] + back_mean - loop_mean), 0, 255)

            img[(mask_tran > 0)[:, :, ch]] = img_tran[(mask_tran > 0)[:, :, ch]]

        # 平滑操作
        img_patch_smooth = cv2.blur(img, (8, 8))
        mask_tran_total = mask_tran.sum(-1).astype('uint8')
        kernel = np.ones((3, 3), np.uint8)
        erosion = cv2.erode(mask_tran_total, kernel, iterations=1)
        dilation = cv2.dilate(mask_tran_total, kernel, iterations=1)
        smooth_loop = dilation - erosion
        img[smooth_loop > 0] = img_patch_smooth[smooth_loop > 0]

        # add tran and ori
        mask = mask + mask_tran

    return img,mask


class TrainvalDataset(Dataset):
    def __init__(self, df, data_folder, mean, std, phase):
        self.df = df
        self.root = data_folder
        self.mean = mean
        self.std = std
        self.phase = phase
        self.aug_dannis = True  # <<<<<<<<<< set by Zhang Ge
        self.transforms = get_transforms(phase, mean, std)
        self.fnames = self.df.index.tolist()



    def __getitem__(self, idx):
        idx = idx if isinstance(idx,int) else idx.item()
        image_id, mask = make_mask(idx, self.df)
        image_path = os.path.join(self.root, "train_images", image_id)
        img = jpeg.JPEG(image_path).decode() #speed up load picture
        #img = cv2.imread(image_path)

        '''if self.aug_dannis:
            img = img_possess(img)'''


        #add mask patch
        aug_patch = False  #<<<<<<<<<<<<<<<<TODO: SET AUG_PATCH
        patch_prob = 0.5
        seed = random.random() #0~1
        if aug_patch and (seed < patch_prob) and self.phase == 'train':
            img = img[:,:,0] #[h,w]

            img, mask = aug_patch_smooth(img, mask)
            # (256, 256) (256, 256, 4)

            # recover img to [h,w,3]
            img = np.expand_dims(img, axis=-1)
            img = np.concatenate((img, img, img), axis=-1) #[256,256,3]

        # if self.phase == 'train':
        #     img, mask = compress(img, mask)
        augmented = self.transforms(image=img, mask=mask)  # [256,256,3],[256,256,4]
        img = augmented['image'].squeeze().astype(np.float32)  # [256,256,3]
        mask = augmented['mask'].squeeze().astype(np.float32)  # [256,256,4]

        '''  debug
        cv2.imwrite("save/" + str(idx) + ".jpg", img)
        mask_save = mask.sum(-1)*255
        cv2.imwrite("save/" + str(idx) + "_mask.jpg", mask_save)
        '''

        #[256,256,3]->[3,256,256]
        img = np.transpose(img,(2,0,1))

        #added by Zhange Ge: get rid of noise
        if self.aug_dannis and self.phase == 'train':
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            for ch in range(mask.shape[-1]):
                mask[:,:,ch] = cv2.morphologyEx(mask[:,:,ch], cv2.MORPH_OPEN, kernel)

        # img[3,256,256]
        mask = mask.transpose((2,0,1))# [4,256,256]
        third_only = False  #<<<<<<<<<<<<<<<<<<<<<<<<<<TODO
        if third_only:
            mask = mask[2, :, :]
            mask = np.expand_dims(mask,axis=0)
            assert mask.shape == (1,256, 256)

        return img, mask

    def __len__(self):
        return len(self.fnames)



class TestDataset(Dataset):
    '''Dataset for test prediction'''
    def __init__(self, root, df, mean, std):
        self.root = root
        df['ImageId'] = df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])
        self.fnames = df['ImageId'].unique().tolist()
        self.aug_dannis = True  # <<<<<<<<<< set by Zhang Ge
        self.num_samples = len(self.fnames)
        self.transform = get_transforms(phase='test', mean=mean, std=std)

    def __getitem__(self, idx):
        idx = idx if isinstance(idx, int) else idx.item()
        fname = self.fnames[idx]
        path = os.path.join(self.root, fname)
        image = cv2.imread(path)
        '''if self.aug_dannis:
            image = img_possess(image)'''
        image = self.transform(image=image)["image"] #[h,w,c]
        image = image.transpose((2,0,1)) #np.array
        return fname, image

    def __len__(self):
        return self.num_samples



def get_transforms(phase, mean, std):
    list_transforms = []
    if phase == "train":
        list_transforms.extend([
                HorizontalFlip(),
                VerticalFlip(),
                RandomBrightnessContrast(p=0.5,brightness_limit=0.5,contrast_limit=0.2),
                GaussNoise(p=0.5), #,var_limit=20.
                #ShiftScaleRotate(p=0.5, shift_limit=0.2, scale_limit=0.2,
                                 #rotate_limit=0, border_mode=cv2.BORDER_CONSTANT)
                #RandomScale(), #problem
                #ElasticTransform()
            ])
    list_transforms.extend([Normalize(mean=mean, std=std, p=1)])

    list_trfms = Compose(list_transforms)
    return list_trfms


def compress(image,mask):
    select_number = random.random()
    dip1 = dip2 = 0
    if select_number < 0.6:
        return image,mask
    elif select_number>0.8:
        dip1 = 30*random.random()
    else:
        dip2 = 30*random.random()
    pts1 = np.float32([[15, 15], [250, 15], [15, 250]])
    pts2 = np.float32([[15 + dip1, 15], [250 - dip2, 15], [15 + dip1, 250]])
    #dip = 25*random.random()
    #pts2 = np.float32([[50, 50 + dip], [1500, 50 + dip], [50, 200 - dip]])
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, (256, 256))
    mask = cv2.warpAffine(mask, M, (256, 256))
    #cv2.imwrite('debug/'+str(select_number*10000)+'.jpg',image)  <<<debug
    return image,mask


def provider(
        data_folder,
        df_path,
        phase,
        fold=0,
        mean=None,
        std=None,
        batch_size=8,
        num_workers=4,
):
    '''Returns dataloader for the model training'''
    df = pd.read_csv(df_path)
    # some preprocessing
    # https://www.kaggle.com/amanooo/defect-detection-starter-u-net
    df['ImageId'], df['ClassId'] = zip(*df['ImageId_ClassId'].str.split('.jpg_')) #e.g. 8648b2010_4.jpg_1
    df['ImageId'] += '.jpg'
    df['ClassId'] = df['ClassId'].astype(int)
    df = df.pivot(index='ImageId', columns='ClassId', values='EncodedPixels')
    df['defects'] = df.count(axis=1)

    #used for cross validation, dataset split split into 5 folds here
    #train_df, val_df = train_test_split(df, test_size=0.1, stratify=df["defects"])
    num_data = df.shape[0]
    df['group'] = np.random.randint(5,size=num_data)
    train_df = df[df['group']!=fold]
    val_df = df[df['group']==fold]

    # added by zhang ge :when training, delete rows that have no RLE data
    #df = train_df[train_df['defects']>0] if phase == "train" else val_df[val_df['defects']>0] #
    df = train_df if phase == "train" else val_df #<<<<<<<<<<<<< TODO:whether only to sample defective
    image_dataset = TrainvalDataset(df, data_folder, mean, std, phase)

    #define sampler:weights of each class:[2.2,14,52,2.5,16]
    resample = False  #<<<<<<<<<<<<<<<<<<<<<<< TODO: SET RESAMPLE
    if phase == 'train' and resample:
        print("Defining Sampler......")

        class_weights = torch.Tensor([2.2,14,52,2.5,16]).cuda()  #to gpu
        #class_weights = torch.sqrt(class_weights)
        print(class_weights)

        size = len(image_dataset)
        sample_targets = torch.zeros(size,dtype=torch.int64).cuda() # to gpu
        for idx in tqdm(range(size),total=size):
            _, mask = make_mask(idx, df) #[256,256,4]
            sum_cls = mask.sum(0).sum(0)
            if(sum_cls.sum()==0): #bg
                sample_targets[idx] = 0
            else: #fg
                #select order: 2,4,1,3->sum_cls:1,3,0,2
                if sum_cls[1]!=0 : sample_targets[idx] = 2
                elif sum_cls[3]!=0 : sample_targets[idx] = 4
                elif sum_cls[0] != 0: sample_targets[idx] = 1
                elif sum_cls[2]!=0 : sample_targets[idx] = 3
        sample_weights = class_weights[sample_targets]
        assert sample_weights.shape[0] == size
        sampler = WeightedRandomSampler(weights=sample_weights,num_samples=size)

    elif phase == 'train' and not resample:
        sampler = RandomSampler(df)
    else:
        #phase != 'train'
        sampler = SequentialSampler(df)

    dataloader = DataLoader(
        image_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
    )

    return dataloader


if __name__ == '__main__':
    pass
    '''
    dataloader = provider(
        data_folder="input/severstal-steel-defect-detection/",
        df_path='input/severstal-steel-defect-detection/train.csv',
        phase='train',
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        batch_size=4,
        num_workers=0,
    )
    for batch in dataloader:
        images, targets = batch
        print(images.shape,targets.shape)
        break
    '''
