from __future__ import print_function, division
import os
import torch
import h5py
import pandas as pd
from skimage import transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from data.sample_data import show_data


class ToFDataset(Dataset):
    def __init__(self, parent_path, transform=None):
        self.parent_path = parent_path
        self.transform = transform
        self.files = []
        for file in os.listdir(self.parent_path):
            data_path = os.path.join(self.parent_path, file)
            if os.path.isfile(data_path) and os.path.splitext(data_path)[1] == '.npz':
                self.files.append(data_path)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path = self.files[idx]
        with np.load(img_path) as data:
            sense_40MHz_img = data.get('arr_0').T
            sense_70MHz_img = data.get('arr_1').T
            depth_1024 = data.get('arr_2').T
            cap_img = np.array([sense_40MHz_img[2,:,:]-sense_40MHz_img[3,:,:],sense_40MHz_img[0,:,:]-sense_40MHz_img[1,:,:],
                                sense_70MHz_img[2,:,:]-sense_70MHz_img[3,:,:],sense_70MHz_img[0,:,:]-sense_70MHz_img[1,:,:]])
        items = {'cap': cap_img, 'depth_1024': depth_1024, 'path': img_path}

        if self.transform:
            items = self.transform(items)

        return items



if __name__ == '__main__':
    dataset = ToFDataset(parent_path=r"/home/saijo/labwork/local-香川研/pytorchDeepToF/numpy_sensor_data")
    print(dataset.files)
    print(len(dataset))

    # data = next(iter(dataset))
    # show_data(data)
    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=True, num_workers=0)
    for i, batch in enumerate(dataloader):
        print(batch['cap'].shape)
        break