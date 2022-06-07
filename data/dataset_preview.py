from __future__ import print_function, division
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils.four_phases_method import four_phases_cropped_depth
# from utils.compare_npphase import show_gen_image
from data.load_npz_plus_AMCW import ToFDataset_3dim
from data import data_preprocess_plus_AMCW
from models.model_cat_version import Generator
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

# test_dataset_folder = r"datasets/v3-cont-corner-test-1st"
# test_dataset_folder = r"datasets/v3-cont-corner-test-3rd"
# test_dataset_folder = r"datasets/v3-corner-1st"
# test_dataset_folder = r"datasets/v3-corner-3rd"
# test_dataset_folder = r'../datasets/v3-cont-corner-train-3rd-appendix'
test_dataset_folder = r'../datasets/appendix-test'

# test_dataset_folder = r"datasets/panaToF-test"
# eval_folder = 'eval'
# nameprefix = f'v3-train1st-e1500-test1st-no-legend-{title}'
# nameprefix = f'v3-train1st-e1500-test1st-{title}'
# nameprefix = f'v3-train3rd-e1500-test1st-{title}'
# nameprefix = f'v3-train3rd-e1500-test3rd-{title}'
# nameprefix = f'v3-corner3rd-no-legend-{title}'
# nameprefix = f'appendix-v3-corner-{title}'
# nameprefix = f'appendix-test-{title}'

# trained_model_path = r'checkpoints/train-rev2-adv0-tv0-beta07/gen_17999times_model.pt'
# trained_model_path = r'checkpoints/train-rev2-adv0-tv1e-6-beta07/gen_17999times_model.pt'
# trained_model_path = r'checkpoints/train-rev2-adv01-tv0-beta07/gen_17999times_model.pt'
# trained_model_path = r'checkpoints/train-rev2-adv01-tv1e-6-beta07/gen_17999times_model.pt'
# trained_model_path = r'checkpoints/train-nonzero-adv02-tv1e-8-beta08/gen_17999times_model.pt'
# trained_model_path = r'checkpoints/train-nonzero-adv02-tv1e-7-beta09/gen_17999times_model.pt'

# trained_model_path = r'checkpoints/appendix-k7-no-corner-v3-1st-adv015-tv8e-7-beta05/gen_1500times_model.pt'
# trained_model_path = r'checkpoints/v3-3rd-adv015-tv8e-7-beta05/gen_1500times_model.pt'
# trained_model_path = r'checkpoints/v3-1st-adv015-tv8e-7-beta05/gen_1500times_model.pt'

dataset = ToFDataset_3dim(parent_path=test_dataset_folder,
                          # transform=transforms.Compose([
                          #     data_preprocess_plus_AMCW.ToTensor(),
                          #     data_preprocess_plus_AMCW.ConstantCrop(),
                          #     # data_preprocess_3dim.panaToFCrop(),
                          #     data_preprocess_plus_AMCW.ImageNormalize(),
                          # ])
                          )

figure = plt.figure(figsize=(8, 8))
cols, rows = 2, 1
for i in range(len(dataset)):
    # print(dataset[i])
    fig = plt.figure()

    ax1 = fig.add_subplot(1, 2, 1)  # 1行２列の１番目
    ax1.imshow(dataset[i]['AMCW'])
    ax1.set_title("AMCW")

    ax2 = fig.add_subplot(1, 2, 2)  # １行２列の２番目
    ax2.imshow(dataset[i]['depth_1024'])
    ax2.set_title("gt_depth")

    plt.tight_layout()
    plt.show()
    # plt.imshow(img.squeeze(), cmap="gray")
    # plt.show()