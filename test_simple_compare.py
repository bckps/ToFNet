from __future__ import print_function, division
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils.four_phases_method import four_phases_cropped_depth
# from utils.compare_npphase import show_gen_image
from data.load_npz_3dim import ToFDataset_3dim
from data import data_preprocess_3dim
from models.model_cat_version import Generator
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

batch_size = 1
# Each image is 256x256 in size
IMG_WIDTH = 256
IMG_HEIGHT = 256
INPUT_CHANNELS = 3
OUTPUT_CHANNELS = 1

title = 'L1+adv+TV'
phase_offset = 0

"""
show_gen_imageはグラフを作るだけの関数です。
必要がない場合はたたんでください。
"""

def show_gen_image(results_folder_path, image_prefix, depth_image, gen_image,z_corr, iters):

    # retrieve distance from normarized depth map

    depth_img = 6.6 * (depth_image[0, 0, :, :].detach().cpu().numpy() + 1) / 2
    gen_img = 6.6 * (gen_image[0, 0, :, :].detach().cpu().numpy() + 1) / 2
    z_corr = z_corr[0, :, :]
    z_corr = np.where(z_corr - phase_offset < 0, 0, z_corr - phase_offset)


    """
    GT: depth_imgのグラフ
    """
    fig, axs = plt.subplots(2, 1,figsize=(4,9))
    plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    plt.subplots_adjust(wspace=0.1,hspace=0.1)
    im0 = axs[0].imshow(depth_img)
    axs[0].set_title(title, fontsize=28)
    axs[0].hlines([32], 0, 127, "red", linestyles='dashed',linewidth = 4)
    axs[0].set_xlim([0, 127])
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[1].plot(depth_img[128 // 4, :], label="ground_truth")
    axs[1].plot(z_corr[128 // 4, :], label='ratio')
    axs[1].plot(gen_img[128 // 4, :], label="ToFNet")
    axs[1].set_xlabel('Column number', fontsize=18)
    axs[1].tick_params(axis='x', labelsize=16)
    axs[1].tick_params(axis='y', labelsize=16)
    fig.savefig(os.path.join(results_folder_path, 'row32th_gt_'+title+'_'+image_prefix+'.png'))
    plt.close()


    """
    パルス変調による距離: z_corrのグラフ
    """
    fig, axs = plt.subplots(2, 1,figsize=(4,9))
    plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    plt.subplots_adjust(wspace=0.1,hspace=0.1)
    im0 = axs[0].imshow(z_corr)
    axs[0].set_title('z_corr', fontsize=28)
    axs[0].hlines([32], 0, 127, "red", linestyles='dashed',linewidth = 4)
    axs[0].set_xlim([0, 127])
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    im0 = axs[1].imshow(gen_img)
    axs[1].set_title('gen_img', fontsize=28)
    axs[1].hlines([32], 0, 127, "red", linestyles='dashed',linewidth = 4)
    axs[1].set_xlim([0, 127])
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    fig.savefig(os.path.join(results_folder_path, 'row32th_randTofnet_gt_'+title+'_'+image_prefix+'.png'))
    plt.close()


    """
    ToFNetによる距離: gen_imgのグラフ
    """
    fig, axs = plt.subplots(2, 1,figsize=(4,9))
    plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    plt.subplots_adjust(wspace=0.1,hspace=0.1)
    im0 = axs[0].imshow(gen_img)
    axs[0].set_title('gen_img', fontsize=28)
    axs[0].hlines([32], 0, 127, "red", linestyles='dashed',linewidth = 4)
    axs[0].set_xlim([0, 127])
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[1].plot(depth_img[128 // 4, :], label="ground_truth")
    axs[1].plot(gen_img[128 // 4, :], label=title)
    axs[1].legend(fontsize=14)
    axs[1].set_xlabel('Column number', fontsize=18)
    axs[1].tick_params(axis='x', labelsize=16)
    axs[1].tick_params(axis='y', labelsize=16)
    fig.savefig(os.path.join(results_folder_path, 'row32th_'+title+'_'+image_prefix+'.png'))
    plt.close()


    """
    GTとパルス変調の距離の比較: depth_imgとz_corrの比較
    """
    fig, axs = plt.subplots(2, 1,figsize=(4,9))
    plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    plt.subplots_adjust(wspace=0.1,hspace=0.1)
    im0 = axs[0].imshow(z_corr)
    axs[0].set_title('ratio method', fontsize=28)
    axs[0].hlines([32], 0, 127, "red", linestyles='dashed',linewidth = 4)
    axs[0].set_xlim([0, 127])
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[1].plot(depth_img[128 // 4, :], label="ground_truth")
    axs[1].plot(z_corr[128 // 4, :], label='ratio')
    axs[1].legend(fontsize=14)
    axs[1].set_xlabel('Column number', fontsize=18)
    axs[1].tick_params(axis='x', labelsize=16)
    axs[1].tick_params(axis='y', labelsize=16)
    fig.savefig(os.path.join(results_folder_path, 'row32th_'+title+'_'+image_prefix+'_phase.png'))
    plt.close()


    """
    GTとToFNetの距離の比較: depth_imgとToFNetの比較(差)
    """
    fig, axs = plt.subplots(2, 1,figsize=(4,9))
    plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    plt.subplots_adjust(wspace=0.1,hspace=0.1)
    im0 = axs[0].imshow(gen_img)
    axs[0].set_title('gen_img', fontsize=28)
    axs[0].set_xlim([0, 127])
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    im0 = axs[1].imshow(np.abs(gen_img-depth_img))
    axs[1].set_title('diff', fontsize=28)
    axs[1].set_xlim([0, 127])
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    fig.savefig(os.path.join(results_folder_path, 'row32th_'+title+'_'+image_prefix+'diff.png'))
    plt.close()


    """
    GTとパルス変調の距離の比較: depth_imgとz_corrの比較(差)
    """
    fig, axs = plt.subplots(2, 1,figsize=(4,9))
    plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    plt.subplots_adjust(wspace=0.1,hspace=0.1)
    im0 = axs[0].imshow(z_corr)
    axs[0].set_title('ratio method', fontsize=28)
    axs[0].set_xlim([0, 127])
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    im0 = axs[1].imshow(np.abs(z_corr-depth_img))
    axs[1].set_title('diff', fontsize=28)
    axs[1].set_xlim([0, 127])
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    fig.savefig(os.path.join(results_folder_path, 'row32th_'+title+'_'+image_prefix+'_phase_diff.png'))
    plt.close()


    """
    GT: depth_img
    """
    fig, axs = plt.subplots()
    im0 = axs.imshow(depth_img)
    axs.set_title('GT', fontsize=28)
    axs.hlines([32], 0, 127, "red", linestyles='dashed',linewidth = 4)
    axs.set_xlim([0, 127])
    axs.set_xticks([])
    axs.set_yticks([])
    plt.close()


    """
    GTとパルス変調の距離の比較: depth_imgとz_corrの比較(128 // 4, 128*2 // 4, 128*3 // 4)
    """
    fig, axs = plt.subplots(4, 1,figsize=(4,18))
    plt.subplots_adjust(wspace=0.1,hspace=0)
    im0 = axs[0].imshow(z_corr)
    axs[0].set_title('Ratio', fontsize=18)
    axs[0].hlines([32], 0, 128, "red", linestyles='dashed',linewidth = 4)
    axs[0].hlines([64], 0, 128, "green", linestyles='dashed',linewidth = 4)
    axs[0].hlines([96], 0, 128, "blue", linestyles='dashed',linewidth = 4)
    axs[0].set_xlim([0, 128])
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[1].plot(depth_img[128 // 4, :], label="ground_truth")
    axs[1].plot(z_corr[128 // 4, :], label="ratio_depth")
    axs[1].legend(fontsize=14)
    axs[2].plot(depth_img[128 // 2, :], label="ground_truth")
    axs[2].plot(z_corr[128 // 2, :], label="ratio_depth")
    axs[2].legend()
    axs[3].plot(depth_img[128*3 // 4, :], label="ground_truth")
    axs[3].plot(z_corr[128*3 // 4, :], label="ratio_depth")
    axs[3].legend()
    axs[3].set_xlabel('Column number', fontsize=18)
    plt.close()

if __name__ == '__main__':

    test_dataset_folder = r"datasets/v3-corner-3rd"
    eval_folder = 'eval'
    nameprefix = f'appendix2-ToFNet2-corner'
    trained_model_path = r'checkpoints/appendix2-corner-v3-3rd-adv015-tv8e-7-beta05/gen_1500times_model.pt'

    dataset = ToFDataset_3dim(parent_path=test_dataset_folder,
                              transform=transforms.Compose([
                                  data_preprocess_3dim.ToTensor(),
                                  data_preprocess_3dim.ConstantCrop(),
                                  # data_preprocess_3dim.panaToFCrop(),
                                  data_preprocess_3dim.ImageNormalize(),
                              ]))
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=1,worker_init_fn = lambda id: np.random.seed(42))

    results_folder_path = os.path.join(eval_folder,nameprefix)
    os.makedirs(results_folder_path, exist_ok=True)
    generator = Generator(in_nc=INPUT_CHANNELS)
    generator.load_state_dict(torch.load(trained_model_path)['model_state_dict'])
    generator.eval()

    for i_batch, sample_batched in enumerate(dataloader):
        cap_img, depth_img, z_corr, data_path = sample_batched['cap'],  \
                                            sample_batched['depth_1024'], sample_batched['phase'][0], sample_batched['path']

        fname, _ = os.path.splitext(os.path.basename(data_path[0]))
        gen_batch = generator(cap_img)
        print(np.min(cap_img.numpy()),np.max(cap_img.numpy()))
        # plt.imshow(cap_img[0,0,:,:])
        # plt.show()
        show_gen_image(results_folder_path,fname, depth_img, gen_batch,z_corr, i_batch)
        print(i_batch)