from __future__ import print_function, division
import os, re
import torch
import h5py
import pandas as pd
from skimage import transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from data.sample_data import show_data


class ToFDataset_3dim(Dataset):
    def __init__(self, parent_path, transform=None):
        self.parent_path = parent_path
        self.transform = transform
        self.filesplit = re.compile('[-\.]')
        self.files = []
        for file in os.listdir(self.parent_path):
            data_paths = [os.path.join(self.parent_path, file, f) for f in os.listdir(os.path.join(self.parent_path, file))]
            self.files.append(data_paths)
#os.path.splitext(data_path)[1]
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_paths = self.files[idx]
        for impath in img_paths:
            # print(re.split(self.filesplit, impath))
            op = re.split(self.filesplit, impath)
            if op[-2]=='A0':
                with np.load(impath) as data: A0_img = data.get('arr_0')
            elif op[-2]=='A1':
                with np.load(impath) as data: A1_img = data.get('arr_0')
            elif op[-2]=='A2':
                with np.load(impath) as data: A2_img = data.get('arr_0')
            elif op[-2]=='truth':
                with np.load(impath) as data: depth_1024 = data.get('arr_0')
            elif op[-2]=='depth':
                with np.load(impath) as data: phase_img = data.get('arr_0')
        cap_img = np.array([A0_img,A1_img,A2_img])
        items = {'cap': cap_img, 'depth_1024': depth_1024, 'phase':phase_img, 'path': ('-').join(op[:-2])}

        if self.transform:
            items = self.transform(items)

        return items



if __name__ == '__main__':
    dataset = ToFDataset_3dim(parent_path=r"/home/saijo/labwork/pytorchDToF-edit-panaToF/datasets/train_3dim")
    # print(dataset.files)
    # print(len(dataset))

    # data = next(iter(dataset))
    # show_data(data)
    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=True, num_workers=0)
    mlist = []
    for i, batch in enumerate(dataloader):
        # print(batch['cap'].shape)
        # print(torch.max(batch['cap']))
        mlist.append(torch.max(batch['cap']).data)
        print(batch['path'])
    print(max(mlist))