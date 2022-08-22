import os
import pandas as pd
# from PIL import Image
import cv2
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from typing import Optional

class Image_Dataset_infer(Dataset):
    def __init__(self, dir_img, H=256, W=256):
        self.dir_img = dir_img
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((int(H), int(W))),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
        _, _, imgs = list(os.walk(self.dir_img))[0]
        imgs = [x for x in imgs if x.split(".")[-1] in ["jpg", "JPG", "png", "PNG"]]
        imgs.sort()
        self.df = pd.DataFrame(imgs)

    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        path_img = self.df.values[idx, 0]
        path_img = os.path.join(self.dir_img, path_img)
        image = cv2.imread(path_img)
        # convert the color
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        return image


class Image_Dataset(Dataset):
    def __init__(self, dir_base, train_val=None, transform=None):
        self.dir_base = dir_base
        self.train_val = train_val
        self.transform = transform
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor()])
        str_dataset = self.dir_base.split("/")[-1]
        if str_dataset in ["large_fine_food_iccv"]:
            subdir = ["Train", "Val", "Val"]
        elif str_dataset in ["CUB_200_2011", "102flowers"]:
            subdir = ["Train", "Test", "Test"]
        elif str_dataset in ["ETZH-101"]:
            subdir = ["images", "images_val", "images_test"]
        elif str_dataset in ["food-101", "stanford_cars"]:
            subdir = ["train", "test", "test"]
        elif str_dataset in ["vfn"]:
            subdir = ["vfn_train", "vfn_val", "vfn_test"]
        elif str_dataset in ["vfn_2.0"]:
            subdir = ["train", "val", "test"]
        if train_val == "train" or train_val == None:
            df_filename = "label/train.txt"
            self.dir_img = os.path.join(self.dir_base, subdir[0])
        elif train_val == "val":
            df_filename = "label/val.txt"
            self.dir_img = os.path.join(self.dir_base, subdir[1])
        elif train_val == "test":
            df_filename = "label/test.txt"
            self.dir_img = os.path.join(self.dir_base, subdir[2])
        path_file = df_filename # os.path.join(self.dir_base, df_filename)
        self.df = pd.read_csv(path_file, sep = ' ', header=None)
        if train_val == "val":
            self.df = self.df.sample(frac=1).reset_index(drop=True) # shuffle only one time, no need to shuffle in dataloader
    
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        path_img, label = self.df.values[idx, 0], self.df.values[idx, 1]
        path_img = os.path.join(self.dir_img, path_img)
        image = cv2.imread(path_img)
        # convert the color
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        return image, label


class Image_DataModule(pl.LightningDataModule):
    def __init__(self, batch_size, stage=None, dir_data="/pub2/luo333/dataset/large_fine_food_iccv", H=256, W=256):
        super().__init__()
        self.dir_data = dir_data
        self.batch_size = batch_size
        self.stage = stage
        self.transform_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize((int(H)+20, int(W)+20)),
            transforms.RandomCrop((int(H), int(W)), padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
        self.transform_val = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((int(H), int(W))),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
        self.transform_test = self.transform_val
        """ transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]) """

    def prepare_data(self):
        # download online data to local folder; or do nothing with local data
        pass

    def setup(self, stage: Optional[str] = None): # see https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html
        # Assign train/val datasets for use in dataloaders
        self.stage = stage
        if self.stage in (None, 'fit'):
            self.data_train = Image_Dataset(self.dir_data, train_val="train", transform=self.transform_train)
            self.data_val = Image_Dataset(self.dir_data, train_val="val", transform=self.transform_val)
        # Assign test dataset for use in dataloader(s)
        if self.stage in (None, 'test'):
            self.data_test = Image_Dataset(self.dir_data, train_val="test", transform=self.transform_test)
    
    '''
    note: for pytorch 1.9.0
    for Trainer(strategy="ddp_spawn"), we can set pin_memory=True
    for Trainer(strategy="ddp_find_unused_parameters_false"), we can only set pin_memory=False, 
        since, if pin_memory=True, then raise warning: [W pthreadpool-cpp.cc:90] Warning: Leaking Caffe2 thread-pool after fork. (function pthreadpool)
    set
    '''
    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True, num_workers=4, persistent_workers=True, pin_memory=False)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, shuffle=False, num_workers=4, persistent_workers=True, pin_memory=False)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, shuffle=False, num_workers=4, persistent_workers=True, pin_memory=False)


if __name__ == "__main__":
    import numpy as np
    path = "/pub2/luo333/dataset/large_fine_food_iccv"
    H, W = 256, 256
    transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((int(H), int(W))),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
    a = Image_Dataset(path, train_val="train", transform=transform)
    b, c = a[0]
    print(b.cpu().detach().numpy().astype(np.float32))
    print(b.shape)
    print(c)
    print(len(a))