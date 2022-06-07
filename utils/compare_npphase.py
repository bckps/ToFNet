from __future__ import print_function, division
import os
import pathlib
import time
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from skimage import io, transform
# import numpy as np
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.transforms import functional as TF
from matplotlib import pyplot as plt
from torchsummary import summary
from utils.four_phases_method import four_phases_depth,four_phases_cropped_depth
import h5py
from PIL import Image
from data.load_npz import ToFDataset
from data import data_preprocess

from models.model_cat_version import Generator, Discriminator, PatchDiscriminator

batch_size = 1
# Each image is 256x256 in size
IMG_WIDTH = 256
IMG_HEIGHT = 256
INPUT_CHANNELS = 3
OUTPUT_CHANNELS = 1
LAMBDA = 100



def show_gen_image(results_folder_path, image_prefix, depth_image, gen_image,z_corr, iters):

    # retrieve distance from normarized depth map
    # 30x10^(-12): [time per frame] x 1024: [frame] x 3x10^8: [speed of light] x depth_map / 2
    # 3 x 1.024 x 3 x depth_map / 2

    depth_img = 6.6 * (depth_image[0, 0, :, :].detach().cpu().numpy() + 1) / 2
    gen_img = 6.6 * (gen_image[0, 0, :, :].detach().cpu().numpy() + 1) / 2
    z_corr = z_corr[0, :, :]

    fig, axs = plt.subplots(2, 2,figsize=(9,9))
    plt.subplots_adjust(wspace=0.3,hspace=0.3)
    im0 = axs[0,0].imshow(depth_img, vmin=0, vmax=5)
    cbar = fig.colorbar(im0, ax=axs[0,0])
    cbar.ax.set_ylabel('Depth [m]',labelpad=7, rotation=90,fontsize=14)
    axs[0,0].set_title('ground_truth',fontsize=18)
    axs[0,0].set_xlabel('Column number',fontsize=14)
    axs[0, 0].set_ylabel('Row number',labelpad=0,fontsize=14)
    axs[0, 0].hlines([32], 0, 128, "red", linestyles='dashed',linewidth = 4)
    # axs[0, 0].hlines([64], 0, 128, "green", linestyles='dashed',linewidth = 4)
    # axs[0, 0].hlines([96], 0, 128, "blue", linestyles='dashed',linewidth = 4)
    axs[0, 0].set_xlim([0, 128])

    im1 = axs[0,1].imshow(z_corr, vmin=0, vmax=5)
    cbar = fig.colorbar(im1, ax=axs[0,1])
    cbar.ax.set_ylabel('Depth [m]',labelpad=7, rotation=90,fontsize=14)
    axs[0,1].set_title('phase_depth',fontsize=18)
    axs[0,1].set_xlabel('Column number',fontsize=14)
    axs[0, 1].set_ylabel('Row number',labelpad=0,fontsize=14)
    axs[0, 1].hlines([32], 0, 128, "red", linestyles='dashed',linewidth = 4)
    # axs[0, 1].hlines([64], 0, 128, "green", linestyles='dashed',linewidth = 4)
    # axs[0, 1].hlines([96], 0, 128, "blue", linestyles='dashed',linewidth = 4)
    axs[0, 1].set_xlim([0, 128])

    imgen = axs[1,0].imshow(gen_img, vmin=0, vmax=5)
    cbar = fig.colorbar(imgen, ax=axs[1,0])
    cbar.ax.set_ylabel('Depth [m]',labelpad=7, rotation=90,fontsize=14)
    axs[1,0].set_title('deep_learning_depth',fontsize=18)
    axs[1,0].set_xlabel('Column number',fontsize=14)
    axs[1,0].set_ylabel('Row number',labelpad=0,fontsize=14)
    axs[1,0].hlines([32], 0, 128, "red", linestyles='dashed',linewidth = 4)
    # axs[1,0].hlines([64], 0, 128, "green", linestyles='dashed',linewidth = 4)
    # axs[1,0].hlines([96], 0, 128, "blue", linestyles='dashed',linewidth = 4)
    axs[1,0].set_xlim([0, 128])

    axs[1,1].plot(depth_img[128 // 4, :], label="ground_truth")
    axs[1,1].plot(z_corr[128 // 4, :], label="phase_depth")
    axs[1,1].plot(gen_img[128 // 4, :], label="deep_learning_depth")
    axs[1,1].legend()
    axs[1,1].set_title('32th row depth (red line)',fontsize=18)
    axs[1,1].set_xlabel('Column number',fontsize=14)
    axs[1,1].set_ylabel('Depth [m]',fontsize=14)
    fig.savefig(os.path.join(results_folder_path, '{}_row32th_'.format(iters)+image_prefix+'.png'))
    # plt.show()
    plt.close()

    fig, axs = plt.subplots(2, 2,figsize=(9,9))
    plt.subplots_adjust(wspace=0.3,hspace=0.3)
    im0 = axs[0,0].imshow(depth_img, vmin=0, vmax=5)
    cbar = fig.colorbar(im0, ax=axs[0,0])
    cbar.ax.set_ylabel('Depth [m]',labelpad=7, rotation=90,fontsize=14)
    axs[0,0].set_title('ground_truth',fontsize=18)
    axs[0,0].set_xlabel('Column number',fontsize=14)
    axs[0, 0].set_ylabel('Row number',labelpad=0,fontsize=14)
    axs[0,0].hlines([64], 0, 128, "green", linestyles='dashed',linewidth = 4)
    # axs[0, 0].hlines([64], 0, 128, "green", linestyles='dashed',linewidth = 4)
    # axs[0, 0].hlines([96], 0, 128, "blue", linestyles='dashed',linewidth = 4)
    axs[0, 0].set_xlim([0, 128])

    im1 = axs[0,1].imshow(z_corr, vmin=0, vmax=5)
    cbar = fig.colorbar(im1, ax=axs[0,1])
    cbar.ax.set_ylabel('Depth [m]',labelpad=7, rotation=90,fontsize=14)
    axs[0,1].set_title('phase_depth',fontsize=18)
    axs[0,1].set_xlabel('Column number',fontsize=14)
    axs[0, 1].set_ylabel('Row number',labelpad=0,fontsize=14)
    axs[0,1].hlines([64], 0, 128, "green", linestyles='dashed',linewidth = 4)
    # axs[0, 1].hlines([64], 0, 128, "green", linestyles='dashed',linewidth = 4)
    # axs[0, 1].hlines([96], 0, 128, "blue", linestyles='dashed',linewidth = 4)
    axs[0, 1].set_xlim([0, 128])

    imgen = axs[1,0].imshow(gen_img, vmin=0, vmax=5)
    cbar = fig.colorbar(imgen, ax=axs[1,0])
    cbar.ax.set_ylabel('Depth [m]',labelpad=7, rotation=90,fontsize=14)
    axs[1,0].set_title('deep_learning_depth',fontsize=18)
    axs[1,0].set_xlabel('Column number',fontsize=14)
    axs[1,0].set_ylabel('Row number',labelpad=0,fontsize=14)
    # axs[1,0].hlines([32], 0, 128, "red", linestyles='dashed',linewidth = 4)
    axs[1,0].hlines([64], 0, 128, "green", linestyles='dashed',linewidth = 4)
    # axs[1,0].hlines([96], 0, 128, "blue", linestyles='dashed',linewidth = 4)
    axs[1,0].set_xlim([0, 128])

    axs[1,1].plot(depth_img[128 // 2, :], label="ground_truth")
    axs[1,1].plot(z_corr[128 // 2, :], label="phase_depth")
    axs[1,1].plot(gen_img[128 // 2, :], label="deep_learning_depth")
    axs[1,1].legend()
    axs[1,1].set_title('64th row depth (green line)',fontsize=18)
    axs[1,1].set_xlabel('Column number',fontsize=14)
    axs[1,1].set_ylabel('Depth [m]',fontsize=14)
    fig.savefig(os.path.join(results_folder_path,'{}_row64th_'.format(iters)+image_prefix+'.png'))
    # plt.show()
    plt.close()

    fig, axs = plt.subplots(2, 2,figsize=(9,9))
    plt.subplots_adjust(wspace=0.3,hspace=0.3)
    im0 = axs[0,0].imshow(depth_img, vmin=0, vmax=5)
    cbar = fig.colorbar(im0, ax=axs[0,0])
    cbar.ax.set_ylabel('Depth [m]',labelpad=7, rotation=90,fontsize=14)
    axs[0,0].set_title('ground_truth',fontsize=18)
    axs[0,0].set_xlabel('Column number',fontsize=14)
    axs[0, 0].set_ylabel('Row number',labelpad=0,fontsize=14)
    axs[0,0].hlines([96], 0, 128, "blue", linestyles='dashed',linewidth = 4)
    # axs[0, 0].hlines([64], 0, 128, "green", linestyles='dashed',linewidth = 4)
    # axs[0, 0].hlines([96], 0, 128, "blue", linestyles='dashed',linewidth = 4)
    axs[0, 0].set_xlim([0, 128])

    im1 = axs[0,1].imshow(z_corr, vmin=0, vmax=5)
    cbar = fig.colorbar(im1, ax=axs[0,1])
    cbar.ax.set_ylabel('Depth [m]',labelpad=7, rotation=90,fontsize=14)
    axs[0,1].set_title('phase_depth',fontsize=18)
    axs[0,1].set_xlabel('Column number',fontsize=14)
    axs[0, 1].set_ylabel('Row number',labelpad=0,fontsize=14)
    axs[0,1].hlines([96], 0, 128, "blue", linestyles='dashed',linewidth = 4)
    # axs[0, 1].hlines([64], 0, 128, "green", linestyles='dashed',linewidth = 4)
    # axs[0, 1].hlines([96], 0, 128, "blue", linestyles='dashed',linewidth = 4)
    axs[0, 1].set_xlim([0, 128])

    imgen = axs[1,0].imshow(gen_img, vmin=0, vmax=5)
    cbar = fig.colorbar(imgen, ax=axs[1,0])
    cbar.ax.set_ylabel('Depth [m]',labelpad=7, rotation=90,fontsize=14)
    axs[1,0].set_title('deep_learning_depth',fontsize=18)
    axs[1,0].set_xlabel('Column number',fontsize=14)
    axs[1,0].set_ylabel('Row number',labelpad=0,fontsize=14)
    # axs[1,0].hlines([32], 0, 128, "red", linestyles='dashed',linewidth = 4)
    # axs[1,0].hlines([64], 0, 128, "green", linestyles='dashed',linewidth = 4)
    axs[1,0].hlines([96], 0, 128, "blue", linestyles='dashed',linewidth = 4)
    axs[1,0].set_xlim([0, 128])

    axs[1,1].plot(depth_img[128*3 // 4, :], label="ground_truth")
    axs[1,1].plot(z_corr[128*3 // 4, :], label="phase_depth")
    axs[1,1].plot(gen_img[128*3 // 4, :], label="deep_learning_depth")
    axs[1,1].legend()
    axs[1,1].set_title('96th row depth (blue line)',fontsize=18)
    axs[1,1].set_xlabel('Column number',fontsize=14)
    axs[1,1].set_ylabel('Depth [m]',fontsize=14)
    fig.savefig(os.path.join(results_folder_path,'{}_row96th_'.format(iters)+image_prefix+'.png'))
    # plt.show()
    plt.close()


if __name__ == '__main__':
    dataset = ToFDataset(parent_path=r"/home/saijo/labwork/local-香川研/pytorchDeepToF/numpy_sensor_data",
                              transform=transforms.Compose([
                                  data_preprocess.ToTensor(),
                                  data_preprocess.ConstantCrop(),
                                  data_preprocess.ImageNormalize(),
                              ]))
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=0,worker_init_fn = lambda id: np.random.seed(42))


    eval_folder = '/home/saijo/labwork/pytorchDToF-edit/eval'
    nameprefix = 'sample_cat_random_decay2'
    results_folder_path = os.path.join(eval_folder,nameprefix)
    os.makedirs(results_folder_path, exist_ok=True)
    generator = Generator()
    generator.load_state_dict(torch.load(r'/home/saijo/labwork/pytorchDToF-edit/checkpoints/sample_cat_random_decay2/gen_39999times_model.pt')['model_state_dict'])
    generator.eval()


    for i_batch, sample_batched in enumerate(dataloader):
        cap_img, depth_img, data_path = sample_batched['cap'],  \
                                            sample_batched['depth_1024'],sample_batched['path']


        fname, _ = os.path.splitext(os.path.basename(data_path[0]))
        # gen_batch = generator(cap_img)
        gen_batch = generator(cap_img)
        sample_batched['gen'] = gen_batch


        z_corr = four_phases_cropped_depth(cap_img[0,:,:,:], datapath=data_path[0])
        show_gen_image(results_folder_path,fname, cap_img[0,:,:,:], depth_img, gen_batch,z_corr, i_batch)
        print(i_batch)
        # observe 4th batch and stop.
        # if i_batch == 3:+
        #     break