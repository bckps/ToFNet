from __future__ import print_function, division
import os
import torch
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
from data.sample_data import show_data


class ToFDataset(Dataset):
    def __init__(self, parent_path, transform=None):
        self.parent_path = parent_path
        self.transform = transform
        self.files = []
        for scene in os.listdir(self.parent_path):
            for file in os.listdir(os.path.join(self.parent_path,scene)):
                path = os.path.join(self.parent_path, scene, file)
                if os.path.isfile(path):
                    self.files.append(path)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path = self.files[idx]
        with h5py.File(img_path) as imgs:
            cap_img = np.array(imgs['cap'])
            cap_ideal_img = np.array(imgs['cap_ideal'])
            depth_img = np.array(imgs['depth_1024'])
        items = {'cap': cap_img, 'cap_ideal': cap_ideal_img, 'depth_1024': depth_img, 'path': img_path}

        if self.transform:
            items = self.transform(items)

        return items



if __name__ == '__main__':

    dataset = ToFDataset(parent_path=r"/home/saijo/labwork/local-香川研/makeToFDataset/dataset_shizuoka")
    print(dataset.files)
    print(len(dataset))

    # data = next(iter(dataset))
    # show_data(data)
    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=True, num_workers=0)
    for i, batch in enumerate(dataloader):
        show_data(batch)
        break