from __future__ import print_function, division
import torch
from torchvision import transforms
from torchvision.transforms import functional as TF
from matplotlib import pyplot as plt
from data.load_npz_3dim import ToFDataset_3dim
from torch.utils.data import Dataset, DataLoader


def normalize_param():
    dataset = ToFDataset_3dim(parent_path=r"/home/saijo/labwork/pytorchDToF-edit-panaToF/datasets/train_3dim")
    # print(dataset.files)
    # print(len(dataset))

    # data = next(iter(dataset))
    # show_data(data)
    dataloader = DataLoader(dataset, batch_size=60,
                            shuffle=True, num_workers=0)
    ch_mean = 0
    ch_var = 0
    count = 0
    maxvalue = 0
    for i, batch in enumerate(dataloader):
        # print(batch['cap'].shape)
        # print(torch.max(batch['cap']))
        ch_mean += torch.mean(batch['cap'], dim=(0,2,3))
        ch_var += torch.var(batch['cap'], dim=(0,2,3))
        #正直・全シーン・各シーン・各データで割れる
        chall_temp = torch.max(batch['cap'])  # ・全シーン
        # chall_temp = torch.max(batch['cap'], dim=(0, 1, 2, 3))  # ・各シーン:データセットのパスを1つのシーンだけにする
        # chall_temp = torch.max(batch['cap'], dim=(   1, 2, 3))  # ・各データ
        count += 1
        if maxvalue < chall_temp:
            maxvalue = chall_temp.item()
    ch_mean /= count
    print(ch_mean.numpy(), ch_var.numpy())
    print(maxvalue)

if __name__ == '__main__':
    normalize_param()
    # mean: [3.11842938 6.24030427 0.68059899] var: [22.14534125  5.73035384  0.21797184]
    # max: 100.19024438899942