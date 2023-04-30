import os
import os.path
import pandas as pd
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.utils.data
from sklearn.model_selection import train_test_split
import albumentations
from albumentations.pytorch import ToTensorV2
import Retinex.retinex as retinex
from facenet_pytorch import MTCNN


PRE__MEAN = [0.5, 0.5, 0.5]
PRE__STD = [0.5, 0.5, 0.5]

mtcnn = MTCNN(select_largest=True, min_face_size=64, post_process=False)

def ApplyWeightedRandomSampler(dataset_csv):
    dataframe = pd.read_csv(dataset_csv) # head: image_path, label
    class_counts = dataframe.label.value_counts()
    sample_weights = [1/class_counts[i] for i in dataframe.label.values]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(dataframe), replacement=True)
    return sampler

def data_concate(data):
    img_bona, img_att = data['img_bona'], data['img_att']
    img_bona_aug, img_att_aug = data['img_bona_aug'], data['img_att_aug']
    label_bona, label_att = data['label_bona'], data['label_att']
    img_bona_path, img_att_path = data['img_bona_path'], data['img_att_path']
    img_path = img_bona_path + img_att_path
    img = torch.cat((img_bona, img_att), dim=0)
    img_aug = torch.cat((img_bona_aug, img_att_aug), dim=0)
    label = torch.cat((label_bona, label_att), dim=0)

    return {'images':img, 'img_aug':img_aug, 'labels':label, 'img_path':img_path}

def msrcr(img, kernel=[10, 20, 30]):
    name = str(np.random.randint(10))
    new_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #cv2.imwrite('./test_ori'+name+'.png', new_img)
    new_img = np.expand_dims(new_img, -1)
    new_img = retinex.automatedMSRCR(new_img, kernel)
    new_img = cv2.cvtColor(new_img[:, :, 0], cv2.COLOR_GRAY2RGB)
    #cv2.imwrite('./test_aug'+name+'.png', new_img)
    return new_img

def clahe(img):
    name = str(np.random.randint(10))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #cv2.imwrite('./test_ori'+name+'.png', img)
    img = np.expand_dims(img, -1)
    cla = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = cla.apply(img)
    cl1 = cv2.cvtColor(cl1, cv2.COLOR_GRAY2RGB)
    #cv2.imwrite('./test_aug'+name+'.png', cl1)
    return cl1

def bi_flow(img, aug='copy'):
    #name = str(np.random.randint(10))
    #cv2.imwrite('./test'+str(name)+'.png', img)
    if aug=='copy':
        img_aug = img.copy()
    elif aug=='msrcr':
        img_aug = msrcr(img)
    elif aug=='clahe':
        #img_aug = clahe(img)
        trans = albumentations.Compose([albumentations.CLAHE()])
        img_aug = trans(image=img)['image']
        #cv2.imwrite('./test_aug_'+str(name)+'.png', img_aug)
    else:
        print('invalid input of bi-flow augmentation')
        img_aug = img.copy()

    img_aug = totensor(image=img_aug)['image']
    return img_aug

def crop_face(img):
    boxes, _ = mtcnn.detect(img)
    try:
        x1, y1, x2, y2 = boxes[0]
        x1, y1 = max(x1, 0), max(y1, 0)
        crop_img = img[int(y1):int(y2), int(x1):int(x2)]
    except: crop_img = None
    return crop_img

totensor = albumentations.Compose([
        albumentations.Normalize(PRE__MEAN, PRE__STD, always_apply=True),
        ToTensorV2()])

def get_test_set(test_data_list, test_csv_list, prop=1.):
    test_video_list = []
    test_label_list = []
    for i in range(len(test_csv_list)):
        data_df = pd.read_csv(test_csv_list[i])
        video = list(data_df['image_path'])
        labels = list(data_df['label'])
        labels = np.array([1 if l=='bonafide' else 0 for l in labels])
        for j in range(len(video)):
            test_video_list.append(os.path.join(test_data_list[i], video[j]))
            test_label_list.append(labels[j])
    
    # random sample
    sample_idx = np.random.choice(len(test_video_list), size=int(prop*len(test_video_list)), replace=False)
    test_videos = np.array(test_video_list)[sample_idx]
    test_labels = np.array(test_label_list)[sample_idx]

    return [test_videos, test_labels]

# =========================================================================================
class EqualSample(Dataset):
    def __init__(self, img_list, label_list, input_shape=(224, 224), bi_flow='copy', train=True, match='min', rnd_seed=42):
        super(EqualSample, self).__init__()
        self.rnd_seed = rnd_seed
        self.bi_flow = bi_flow
        self.train = train
        #self.data_dir = data_dir
        self.img_list = img_list
        #self.label_list = np.array([1 if l=='bonafide' else 0 for l in label_list])
        self.label_list = label_list

        # augmentation
        self.composed_transformations = albumentations.Compose([
            albumentations.HorizontalFlip(),
            #albumentations.RandomGamma(gamma_limit=(80, 180)), # 0.5, 1.5
            albumentations.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20),
            albumentations.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0.1, p=0.5),
            #albumentations.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            albumentations.GaussianBlur(blur_limit=7),
            albumentations.Resize(height=input_shape[0], width=input_shape[1]),
            albumentations.CoarseDropout(max_holes=3, max_height=int(input_shape[0]*0.2), max_width=int(input_shape[1]*0.2), p=0.5)
            #albumentations.RandomGridShuffle(grid=(4, 4), always_apply=False, p=1.0)
            ])

        # label balancing
        self.bona_list_idx = np.argwhere(self.label_list==1).squeeze()
        self.att_list_idx = np.argwhere(self.label_list==0).squeeze()
        if match=='max':
            if len(self.bona_list_idx)>len(self.att_list_idx): self.att_list_idx = self.match_len_max(self.att_list_idx, len(self.bona_list_idx))
            else: self.bona_list_idx = self.match_len_max(self.bona_list_idx, len(self.att_list_idx))
        elif match=='min':
            if len(self.bona_list_idx)>len(self.att_list_idx): self.bona_list_idx = self.match_len_min(self.bona_list_idx, len(self.att_list_idx))
            else: self.att_list_idx = self.match_len_min(self.att_list_idx, len(self.bona_list_idx))
        np.random.shuffle(self.bona_list_idx)
        np.random.shuffle(self.att_list_idx)


    def match_len_max(self, l, length):
        l_full = np.repeat(l, int(length/len(l)))
        np.random.seed(self.rnd_seed)
        l_full = np.concatenate((l_full, (np.random.choice(l, length%len(l), replace=False))))
        return l_full

    def match_len_min(self, l, length):
        np.random.shuffle(l)
        l_short = l[0:length]
        return l_short

    def shuffle(self):
        np.random.shuffle(self.att_list_idx)

    def read_img(self, img_path):
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # augmentation
        image = self.composed_transformations(image = image)['image']

        img_aug = bi_flow(image, self.bi_flow)

        image = totensor(image=image)['image']

        return image, img_aug

    def __len__(self):
        return np.max([len(self.bona_list_idx), len(self.att_list_idx)])

    def __getitem__(self, idx):
        img_bona_path = self.img_list[self.bona_list_idx[idx]]
        img_att_path = self.img_list[self.att_list_idx[idx]]

        img_bona, img_bona_aug = self.read_img(img_bona_path)
        img_att, img_att_aug = self.read_img(img_att_path)
        label_bona = torch.tensor(1, dtype = torch.int64)
        label_att = torch.tensor(0, dtype = torch.int64)

        data_raw = {'img_bona':img_bona, 'img_att':img_att, 
                'img_bona_aug':img_bona_aug, 'img_att_aug':img_att_aug,
                'label_bona':label_bona, 'label_att':label_att, 
                'img_bona_path':img_bona_path, 'img_att_path':img_att_path}

        #data = data_concate(data_raw)
        return data_raw
        

class GridTest(Dataset):
    def __init__(self, img_list, label_list, grid=5, input_shape=(224, 224), bi_flow='copy', crop=True):
        super(GridTest, self).__init__()
        self.image_list = img_list
        #self.label_list = np.array([1 if l=='bonafide' else 0 for l in label_list])
        self.label_list = label_list
        self.crop = crop
        self.grid = grid
        self.bi_flow = bi_flow

        self.grid_transformations = albumentations.Compose([
            albumentations.RandomCrop(height=input_shape[0], width=input_shape[1])])
        self.crop_transformations = albumentations.Compose([
            albumentations.Resize(height=input_shape[0], width=input_shape[1])])

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.crop: 
            image = self.crop_transformations(image = image)['image']
            img_aug = bi_flow(image, self.bi_flow)
            img_aug_list = [img_aug]
            image = totensor(image=image)['image']
            img_list = [image]
        else:
            img_list = []
            img_aug_list = []
            # augmentation
            for _ in range(self.grid):
                image_crop = self.grid_transformations(image = image)['image']
                image_aug = bi_flow(image_crop, self.bi_flow)

                image_crop = totensor(image=image_crop)['image']
                
                img_list.append(image_crop)
                img_aug_list.append(image_aug)
        
        label = self.label_list[idx]

        data = {'images':img_list, 'img_aug':img_aug_list, 'labels':label, 'image_path':img_path}
        return data


if __name__ == "__main__":
    data_dir = r'/sdata/xianyun.sun/SynthASpoof_data'
    sas_csv = r'/sdata/xianyun.sun/SynthASpoof_data/labels.csv'

    sas_df = pd.read_csv(sas_csv)
    sas_img_path = list(sas_df['image_path'])
    sas_label = list(sas_df['label'])

    sas_train, sas_test, sas_label_train, sas_label_test = train_test_split(sas_img_path, sas_label, test_size=0.2, random_state=42)
    train_list = [sas_train, sas_label_train]
    test_list = [sas_test, sas_label_test]

    train_dataset = EqualSample(data_dir, sas_img_path, sas_label, input_shape=(224, 224), bi_flow='clahe')
    train_loader = DataLoader(train_dataset, batch_size=2, pin_memory=True, drop_last=True, shuffle=True)

    for ii, data in enumerate(train_loader):
        print(ii)
        break











