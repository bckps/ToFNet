import numpy as np
import h5py
# import cupy as np
import matplotlib.pyplot as plt
import numpy
import os

import torch


def four_phases_depth(file_path):
    # data = np.load(r'C:\Users\919im\Documents\local-香川研\pytorchDeepToF\numpy_sensor_data\bathroom_1.npz', 'r')
    data = np.load(file_path, 'r')

    sense_40MHz_img = data.get('arr_0')
    sense_70MHz_img = data.get('arr_1')
    depth_1024 = data.get('arr_2')


    z_max = 3e8 / (2*33e6)
    z40_max = 3e8/(2*40e6)
    z70_max = 3e8/(2*70e6)

    depth_1024 = z_max * depth_1024

    phase_40MHz = np.arctan2(sense_40MHz_img[:,:,2]-sense_40MHz_img[:,:,3],sense_40MHz_img[:,:,0]-sense_40MHz_img[:,:,1])
    # phase_40MHz = np.where(phase_40MHz<0,phase_40MHz+2*np.pi,phase_40MHz)
    phase_40MHz = phase_40MHz + np.pi
    # depth_40MHz = z40_max*phase_40MHz/(2*np.pi)

    phase_70MHz = np.arctan2(sense_70MHz_img[:,:,2]-sense_70MHz_img[:,:,3],sense_70MHz_img[:,:,0]-sense_70MHz_img[:,:,1])
    # phase_70MHz = np.where(phase_70MHz<0,phase_70MHz+2*np.pi,phase_70MHz)
    phase_70MHz = phase_70MHz + np.pi
    # depth_70MHz = z70_max*phase_70MHz/(2*np.pi)

    lut_40MHz = np.linspace(0, np.pi * 1024 / 417, num=1024) % (2 * np.pi)
    lut_70MHz = np.linspace(0, np.pi * 1024 / 238, num=1024) % (2 * np.pi)

    error_mat = np.zeros((1024,256,256))
    for i in range(error_mat.shape[0]):
        error_mat[i,:,:] = np.square(phase_40MHz[:,:]-lut_40MHz[i])+np.square(phase_70MHz[:,:]-lut_70MHz[i])
    z_indexes = np.argmin(error_mat, axis=0)
    z_corr = z_max * z_indexes/1024
    # print(z_indexes)

    return z_corr, depth_1024


def four_phases_cropped_depth(cap_img, datapath=None, random=False, save_path='/home/saijo/labwork/pytorchDToF-edit/eval/phase_depth'):

    filename = os.path.splitext(os.path.basename(datapath))[0]

    if random == False:
        for root, dirs, files in os.walk(save_path):
            if filename+'_phase_depth.npz' in files:
                with np.load(os.path.join(save_path, filename+'_phase_depth.npz'), 'r') as data:
                    z_corr = data.get('arr_0')
                print('load success!')
                return z_corr

    if cap_img is not torch.Tensor:
        cap_img = cap_img.to('cpu').detach().numpy().copy()

    _, img_height, img_width = cap_img.shape
    z_max = 3e8 / (2*33e6)
    z40_max = 3e8/(2*40e6)
    z70_max = 3e8/(2*70e6)


    phase_40MHz = np.arctan2(cap_img[0],cap_img[1])
    # phase_40MHz = np.where(phase_40MHz<0,phase_40MHz+2*np.pi,phase_40MHz)
    phase_40MHz = phase_40MHz + np.pi
    # depth_40MHz = z40_max*phase_40MHz/(2*np.pi)

    phase_70MHz = np.arctan2(cap_img[2],cap_img[3])
    # phase_70MHz = np.where(phase_70MHz<0,phase_70MHz+2*np.pi,phase_70MHz)
    phase_70MHz = phase_70MHz + np.pi
    # depth_70MHz = z70_max*phase_70MHz/(2*np.pi)

    lut_40MHz = np.linspace(0, np.pi * 1024 / 417, num=1024) % (2 * np.pi)
    lut_70MHz = np.linspace(0, np.pi * 1024 / 238, num=1024) % (2 * np.pi)

    error_mat = np.zeros((1024,img_height,img_width))
    for i in range(error_mat.shape[0]):
        error_mat[i,:,:] = np.square(phase_40MHz[:,:]-lut_40MHz[i])+np.square(phase_70MHz[:,:]-lut_70MHz[i])
    z_indexes = np.argmin(error_mat, axis=0)
    z_corr = z_max * z_indexes/1024
    # print(z_indexes)

    if filename and (not random):
        with open(os.path.join(save_path, filename+'_phase_depth.npz'), 'wb') as f:
            np.savez(f, z_corr)
    return z_corr

if __name__ == '__main__':

    # print(sense_40MHz_img.shape)
    # fig, axs = plt.subplots(1, 4)
    # axs[0].imshow(sense_40MHz_img[:,:,0])
    # axs[0].set_title('Q1')
    # axs[1].imshow(sense_40MHz_img[:,:,1])
    # axs[1].set_title('Q2')
    # axs[2].imshow(sense_40MHz_img[:,:,2])
    # axs[2].set_title('Q3')
    # axs[3].imshow(sense_40MHz_img[:,:,3])
    # axs[3].set_title('Q4')
    # plt.show()
    #
    # fig, axs = plt.subplots(1, 2)
    # axs[0].imshow(np.where(phase_40MHz>6,1,0))
    # axs[0].set_title('40')
    # axs[1].imshow(np.where(phase_70MHz>6,1,0))
    # axs[1].set_title('70')
    # plt.show()
    z_corr, depth_1024 = four_phases_depth(r'C:\Users\919im\Documents\local-香川研\pytorchDeepToF\numpy_sensor_data\bathroom_1.npz')
    fig, axs = plt.subplots(1, 2)
    im0 = axs[0].imshow(depth_1024, vmin=0, vmax=5)
    fig.colorbar(im0, ax=axs[0])
    axs[0].set_title('depth')
    im1 = axs[1].imshow(z_corr, vmin=0, vmax=5)
    fig.colorbar(im1, ax=axs[1])
    axs[1].set_title('phase_depth')
    fig.savefig(os.path.join(r'C:\Users\919im\Documents\local-香川研\pytorchDeepToF\numpy_sensor_data','four_phase_method.png'))
    plt.show()