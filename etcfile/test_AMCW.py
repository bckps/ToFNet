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

batch_size = 1
# Each image is 256x256 in size
IMG_WIDTH = 256
IMG_HEIGHT = 256
INPUT_CHANNELS = 3
OUTPUT_CHANNELS = 1
LAMBDA = 100

# title = 'L1'
# title = 'L1+adv'
# title = 'L1+TV'
title = 'L1+adv+TV'
phase_offset = 0

def show_gen_image(results_folder_path, image_prefix, depth_image, gen_image,z_corr,AMCW_img, iters):

    # retrieve distance from normarized depth map
    # 30x10^(-12): [time per frame] x 1024: [frame] x 3x10^8: [speed of light] x depth_map / 2
    # 3 x 1.024 x 3 x depth_map / 2

    depth_img = 6.6 * (depth_image[0, 0, :, :].detach().cpu().numpy() + 1) / 2
    gen_img = 6.6 * (gen_image[0, 0, :, :].detach().cpu().numpy() + 1) / 2
    z_corr = z_corr[0, :, :]
    z_corr = np.where(z_corr - phase_offset < 0, 0, z_corr - phase_offset)
    AMCW_img = AMCW_img[0,0, :, :]
    # vmin = 0
    vmin = 2
    # vmax = 5.5
    # diff_vmin = -1
    vmax = 4.2
    # diff_vmax = 1

    fig, axs = plt.subplots(2, 1,figsize=(4,9))
    plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    plt.subplots_adjust(wspace=0.1,hspace=0.1)
    im0 = axs[0].imshow(depth_img)
    # im0 = axs[0].imshow(z_corr, vmin=0, vmax=5)
    # cbar = fig.colorbar(im0, ax=axs[0])
    # cbar.ax.tick_params(labelsize=16)
    # cbar.ax.set_ylabel('Depth [m]',labelpad=7, rotation=90,fontsize=14)
    # axs[0].set_title('L1 + adv',fontsize=18)
    axs[0].set_title(title, fontsize=28)
    # axs[0,0].set_xlabel('Column number',fontsize=14)
    # axs[0, 0].set_ylabel('Row number',labelpad=0,fontsize=14)
    axs[0].hlines([32], 0, 127, "red", linestyles='dashed',linewidth = 4)
    # axs[0, 0].hlines([64], 0, 128, "green", linestyles='dashed',linewidth = 4)
    # axs[0, 0].hlines([96], 0, 128, "blue", linestyles='dashed',linewidth = 4)
    axs[0].set_xlim([0, 127])
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[1].plot(depth_img[128 // 4, :], label="ground_truth")
    axs[1].plot(z_corr[128 // 4, :], label='ratio')
    # axs[1].plot(gen_img[128 // 4, :], label=title)
    axs[1].plot(gen_img[128 // 4, :], label="ToFNet")
    axs[1].plot(AMCW_img[128 // 4, :], label="AMCW")
    axs[1].legend(fontsize=14)
    # axs[1].set_title('32th row depth (red line)',fontsize=18)
    axs[1].set_xlabel('Column number', fontsize=18)
    axs[1].tick_params(axis='x', labelsize=16)
    axs[1].tick_params(axis='y', labelsize=16)
    # ax2 = axs[1].twinx()
    # ax2.set_ylabel('32th row depth (red line)',fontsize=18)
    fig.savefig(os.path.join(results_folder_path, 'row32th_gt_'+title+'_'+image_prefix+'.png'))
    # plt.show()
    plt.close()

    fig, axs = plt.subplots(2, 1,figsize=(4,9))
    plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    plt.subplots_adjust(wspace=0.1,hspace=0.1)
    im0 = axs[0].imshow(z_corr)
    # im0 = axs[0].imshow(z_corr, vmin=0, vmax=5)
    # cbar = fig.colorbar(im0, ax=axs[0])
    # cbar.ax.tick_params(labelsize=16)
    # cbar.ax.set_ylabel('Depth [m]',labelpad=7, rotation=90,fontsize=14)
    # axs[0].set_title('L1 + adv',fontsize=18)
    axs[0].set_title(title, fontsize=28)
    # axs[0,0].set_xlabel('Column number',fontsize=14)
    # axs[0, 0].set_ylabel('Row number',labelpad=0,fontsize=14)
    axs[0].hlines([32], 0, 127, "red", linestyles='dashed',linewidth = 4)
    # axs[0, 0].hlines([64], 0, 128, "green", linestyles='dashed',linewidth = 4)
    # axs[0, 0].hlines([96], 0, 128, "blue", linestyles='dashed',linewidth = 4)
    axs[0].set_xlim([0, 127])
    axs[0].set_xticks([])
    axs[0].set_yticks([])

    im0 = axs[1].imshow(gen_img)
    # im0 = axs[0].imshow(z_corr, vmin=0, vmax=5)
    # cbar = fig.colorbar(im0, ax=axs[0])
    # cbar.ax.tick_params(labelsize=16)
    # cbar.ax.set_ylabel('Depth [m]',labelpad=7, rotation=90,fontsize=14)
    # axs[0].set_title('L1 + adv',fontsize=18)
    axs[1].set_title(title, fontsize=28)
    # axs[0,0].set_xlabel('Column number',fontsize=14)
    # axs[0, 0].set_ylabel('Row number',labelpad=0,fontsize=14)
    axs[1].hlines([32], 0, 127, "red", linestyles='dashed',linewidth = 4)
    # axs[0, 0].hlines([64], 0, 128, "green", linestyles='dashed',linewidth = 4)
    # axs[0, 0].hlines([96], 0, 128, "blue", linestyles='dashed',linewidth = 4)
    axs[1].set_xlim([0, 127])
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    fig.savefig(os.path.join(results_folder_path, 'row32th_randTofnet_gt_'+title+'_'+image_prefix+'.png'))
    # plt.show()
    plt.close()


    fig, axs = plt.subplots(2, 1,figsize=(4,9))
    plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    plt.subplots_adjust(wspace=0.1,hspace=0.1)
    im0 = axs[0].imshow(gen_img)
    # im0 = axs[0].imshow(z_corr, vmin=0, vmax=5)
    # cbar = fig.colorbar(im0, ax=axs[0])
    # cbar.ax.tick_params(labelsize=16)
    # cbar.ax.set_ylabel('Depth [m]',labelpad=7, rotation=90,fontsize=14)
    # axs[0].set_title('L1 + adv',fontsize=18)
    axs[0].set_title(title, fontsize=28)
    # axs[0,0].set_xlabel('Column number',fontsize=14)
    # axs[0, 0].set_ylabel('Row number',labelpad=0,fontsize=14)
    axs[0].hlines([32], 0, 127, "red", linestyles='dashed',linewidth = 4)
    # axs[0, 0].hlines([64], 0, 128, "green", linestyles='dashed',linewidth = 4)
    # axs[0, 0].hlines([96], 0, 128, "blue", linestyles='dashed',linewidth = 4)
    axs[0].set_xlim([0, 127])
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[1].plot(depth_img[128 // 4, :], label="ground_truth")
    # axs[1].plot(z_corr[128 // 4, :], label=title)
    axs[1].plot(gen_img[128 // 4, :], label=title)
    # axs[1].plot(gen_img[128 // 4, :], label="deep_learning_depth")
    axs[1].legend(fontsize=14)
    # axs[1].set_title('32th row depth (red line)',fontsize=18)
    axs[1].set_xlabel('Column number', fontsize=18)
    axs[1].tick_params(axis='x', labelsize=16)
    axs[1].tick_params(axis='y', labelsize=16)
    # ax2 = axs[1].twinx()
    # ax2.set_ylabel('32th row depth (red line)',fontsize=18)
    fig.savefig(os.path.join(results_folder_path, 'row32th_'+title+'_'+image_prefix+'.png'))
    # plt.show()
    plt.close()

    fig, axs = plt.subplots(2, 1,figsize=(4,9))
    plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    plt.subplots_adjust(wspace=0.1,hspace=0.1)
    # im0 = axs[0].imshow(gen_img, vmin=0, vmax=5)
    im0 = axs[0].imshow(z_corr)
    # cbar = fig.colorbar(im0, ax=axs[0])
    # cbar.ax.tick_params(labelsize=16)
    # cbar.ax.set_ylabel('Depth [m]',labelpad=7, rotation=90,fontsize=14)
    # axs[0].set_title('L1 + adv',fontsize=18)
    axs[0].set_title('ratio method', fontsize=28)
    # axs[0,0].set_xlabel('Column number',fontsize=14)
    # axs[0, 0].set_ylabel('Row number',labelpad=0,fontsize=14)
    axs[0].hlines([32], 0, 127, "red", linestyles='dashed',linewidth = 4)
    # axs[0, 0].hlines([64], 0, 128, "green", linestyles='dashed',linewidth = 4)
    # axs[0, 0].hlines([96], 0, 128, "blue", linestyles='dashed',linewidth = 4)
    axs[0].set_xlim([0, 127])
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[1].plot(depth_img[128 // 4, :], label="ground_truth")
    # axs[1].plot(z_corr[128 // 4, :], label=title)
    axs[1].plot(z_corr[128 // 4, :], label='ratio')
    # axs[1].plot(gen_img[128 // 4, :], label="deep_learning_depth")
    axs[1].legend(fontsize=14)
    # axs[1].set_title('32th row depth (red line)',fontsize=18)
    axs[1].set_xlabel('Column number', fontsize=18)
    axs[1].tick_params(axis='x', labelsize=16)
    axs[1].tick_params(axis='y', labelsize=16)
    # ax2 = axs[1].twinx()
    # ax2.set_ylabel('32th row depth (red line)',fontsize=18)
    fig.savefig(os.path.join(results_folder_path, 'row32th_'+title+'_'+image_prefix+'_phase.png'))
    # plt.show()
    plt.close()


    fig, axs = plt.subplots(2, 1,figsize=(4,9))
    plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    plt.subplots_adjust(wspace=0.1,hspace=0.1)
    im0 = axs[0].imshow(gen_img)
    # im0 = axs[0].imshow(z_corr, vmin=0, vmax=5)
    # cbar = fig.colorbar(im0, ax=axs[0])
    # cbar.ax.tick_params(labelsize=16)
    # cbar.ax.set_ylabel('Depth [m]',labelpad=7, rotation=90,fontsize=14)
    # axs[0].set_title('L1 + adv',fontsize=18)
    axs[0].set_title(title, fontsize=28)
    # axs[0,0].set_xlabel('Column number',fontsize=14)
    # axs[0, 0].set_ylabel('Row number',labelpad=0,fontsize=14)
    # axs[0].hlines([32], 0, 128, "red", linestyles='dashed',linewidth = 4)
    # axs[0, 0].hlines([64], 0, 128, "green", linestyles='dashed',linewidth = 4)
    # axs[0, 0].hlines([96], 0, 128, "blue", linestyles='dashed',linewidth = 4)
    axs[0].set_xlim([0, 127])
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    im0 = axs[1].imshow(np.abs(gen_img-depth_img))
    axs[1].set_xlim([0, 127])
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    # cbar = fig.colorbar(im0, ax=axs[1])
    fig.savefig(os.path.join(results_folder_path, 'row32th_'+title+'_'+image_prefix+'diff.png'))
    # plt.show()
    plt.close()

    fig, axs = plt.subplots(2, 1,figsize=(4,9))
    plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    plt.subplots_adjust(wspace=0.1,hspace=0.1)
    # im0 = axs[0].imshow(gen_img, vmin=0, vmax=5)
    im0 = axs[0].imshow(z_corr)
    # cbar = fig.colorbar(im0, ax=axs[0])
    # cbar.ax.tick_params(labelsize=16)
    # cbar.ax.set_ylabel('Depth [m]',labelpad=7, rotation=90,fontsize=14)
    # axs[0].set_title('L1 + adv',fontsize=18)
    axs[0].set_title('ratio method', fontsize=28)
    # axs[0,0].set_xlabel('Column number',fontsize=14)
    # axs[0, 0].set_ylabel('Row number',labelpad=0,fontsize=14)
    # axs[0].hlines([32], 0, 127, "red", linestyles='dashed',linewidth = 4)
    # axs[0, 0].hlines([64], 0, 128, "green", linestyles='dashed',linewidth = 4)
    # axs[0, 0].hlines([96], 0, 128, "blue", linestyles='dashed',linewidth = 4)
    axs[0].set_xlim([0, 127])
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    im0 = axs[1].imshow(np.abs(z_corr-depth_img))
    axs[1].set_xlim([0, 127])
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    fig.savefig(os.path.join(results_folder_path, 'row32th_'+title+'_'+image_prefix+'_phase_diff.png'))
    # cbar = fig.colorbar(im0, ax=axs[1])
    # plt.show()
    plt.close()

    fig, axs = plt.subplots()
    # plt.subplots_adjust(wspace=0.1,hspace=0.1)
    # im0 = axs[0].imshow(gen_img, vmin=0, vmax=5)
    im0 = axs.imshow(depth_img)
    # cbar = fig.colorbar(im0, ax=axs)
    # cbar.ax.tick_params(labelsize=16)
    # cbar.ax.set_ylabel('Depth [m]',labelpad=7, rotation=90,fontsize=14)
    # axs[0].set_title('L1 + adv',fontsize=18)
    axs.set_title('GT', fontsize=28)
    # axs[0,0].set_xlabel('Column number',fontsize=14)
    # axs[0, 0].set_ylabel('Row number',labelpad=0,fontsize=14)
    axs.hlines([32], 0, 127, "red", linestyles='dashed',linewidth = 4)
    # axs[0, 0].hlines([64], 0, 128, "green", linestyles='dashed',linewidth = 4)
    # axs[0, 0].hlines([96], 0, 128, "blue", linestyles='dashed',linewidth = 4)
    axs.set_xlim([0, 127])
    axs.set_xticks([])
    axs.set_yticks([])

    # ax2 = axs[1].twinx()
    # ax2.set_ylabel('32th row depth (red line)',fontsize=18)

    # fig.savefig(os.path.join(results_folder_path, 'row32th_GT'+image_prefix+'.png'))
    # plt.show()
    plt.close()


    fig, axs = plt.subplots(4, 1,figsize=(4,18))
    plt.subplots_adjust(wspace=0.1,hspace=0)
    # im0 = axs[0].imshow(gen_img, vmin=0, vmax=5)
    im0 = axs[0].imshow(z_corr)
    # cbar = fig.colorbar(im0, ax=axs[0])
    # cbar.ax.set_ylabel('Depth [m]',labelpad=7, rotation=90,fontsize=14)
    # axs[0].set_title('L1 + adv',fontsize=18)
    axs[0].set_title('Ratio', fontsize=18)
    # axs[0,0].set_xlabel('Column number',fontsize=14)
    # axs[0, 0].set_ylabel('Row number',labelpad=0,fontsize=14)
    axs[0].hlines([32], 0, 128, "red", linestyles='dashed',linewidth = 4)
    axs[0].hlines([64], 0, 128, "green", linestyles='dashed',linewidth = 4)
    axs[0].hlines([96], 0, 128, "blue", linestyles='dashed',linewidth = 4)
    # axs[0, 0].hlines([64], 0, 128, "green", linestyles='dashed',linewidth = 4)
    # axs[0, 0].hlines([96], 0, 128, "blue", linestyles='dashed',linewidth = 4)
    axs[0].set_xlim([0, 128])
    axs[0].set_xticks([])
    axs[0].set_yticks([])


    axs[1].plot(depth_img[128 // 4, :], label="ground_truth")
    axs[1].plot(z_corr[128 // 4, :], label="ratio_depth")
    # axs[1].plot(gen_img[128 // 4, :], label="deep_learning_depth")

    axs[1].legend(fontsize=14)
    # axs[1].set_title('32th row depth (red line)',fontsize=18)
    # axs[1].set_xlabel('Column number',fontsize=14)
    # axs[1].set_ylabel('Depth [m]',fontsize=14)

    axs[2].plot(depth_img[128 // 2, :], label="ground_truth")
    axs[2].plot(z_corr[128 // 2, :], label="ratio_depth")
    # axs[2].plot(gen_img[128 // 2, :], label="deep_learning_depth")
    axs[2].legend()

    axs[3].plot(depth_img[128*3 // 4, :], label="ground_truth")
    axs[3].plot(z_corr[128*3 // 4, :], label="ratio_depth")
    # axs[3].plot(gen_img[128*3 // 4, :], label="deep_learning_depth")
    axs[3].legend()

    axs[3].set_xlabel('Column number', fontsize=18)

    # fig.savefig(os.path.join(results_folder_path, '{}_row32_64_96th_Phase'.format(iters)+image_prefix+'.png'))
    # plt.show()
    plt.close()


    # for cax in grid.cbar_axes:
    #     cax.toggle_label(False)

    # plt.savefig(os.path.join(npz_folder_path, 'multi-images.png'))
    # plt.show()

if __name__ == '__main__':


    # test_dataset_folder = r"datasets/v3-cont-corner-test-1st"
    # test_dataset_folder = r"datasets/v3-cont-corner-test-3rd"
    # test_dataset_folder = r"datasets/v3-corner-1st"
    # test_dataset_folder = r"datasets/v3-corner-3rd"
    test_dataset_folder = r'datasets/appendix-test'

    # test_dataset_folder = r"datasets/panaToF-test"
    eval_folder = 'eval'
    # nameprefix = f'v3-train1st-e1500-test1st-no-legend-{title}'
    # nameprefix = f'v3-train1st-e1500-test1st-{title}'
    # nameprefix = f'v3-train3rd-e1500-test1st-{title}'
    # nameprefix = f'v3-train3rd-e1500-test3rd-{title}'
    # nameprefix = f'v3-corner3rd-no-legend-{title}'
    # nameprefix = f'appendix-v3-corner-{title}'
    nameprefix = f'appendix-test-{title}'


    # trained_model_path = r'checkpoints/train-rev2-adv0-tv0-beta07/gen_17999times_model.pt'
    # trained_model_path = r'checkpoints/train-rev2-adv0-tv1e-6-beta07/gen_17999times_model.pt'
    # trained_model_path = r'checkpoints/train-rev2-adv01-tv0-beta07/gen_17999times_model.pt'
    # trained_model_path = r'checkpoints/train-rev2-adv01-tv1e-6-beta07/gen_17999times_model.pt'
    # trained_model_path = r'checkpoints/train-nonzero-adv02-tv1e-8-beta08/gen_17999times_model.pt'
    # trained_model_path = r'checkpoints/train-nonzero-adv02-tv1e-7-beta09/gen_17999times_model.pt'

    # trained_model_path = r'checkpoints/appendix-k7-no-corner-v3-1st-adv015-tv8e-7-beta05/gen_1500times_model.pt'
    trained_model_path = r'checkpoints/v3-3rd-adv015-tv8e-7-beta05/gen_1500times_model.pt'
    # trained_model_path = r'checkpoints/v3-1st-adv015-tv8e-7-beta05/gen_1500times_model.pt'

    dataset = ToFDataset_3dim(parent_path=test_dataset_folder,
                              transform=transforms.Compose([
                                  data_preprocess_plus_AMCW.ToTensor(),
                                  data_preprocess_plus_AMCW.ConstantCrop(),
                                  # data_preprocess_3dim.panaToFCrop(),
                                  data_preprocess_plus_AMCW.ImageNormalize(),
                              ]))



    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=1,worker_init_fn = lambda id: np.random.seed(42))

    results_folder_path = os.path.join(eval_folder,nameprefix)
    os.makedirs(results_folder_path, exist_ok=True)
    generator = Generator(in_nc=INPUT_CHANNELS)
    generator.load_state_dict(torch.load(trained_model_path)['model_state_dict'])
    generator.eval()

    for i_batch, sample_batched in enumerate(dataloader):
        cap_img, depth_img, z_corr, data_path, AMCW_img = sample_batched['cap'],  \
                                            sample_batched['depth_1024'], sample_batched['phase'][0], sample_batched['path'], sample_batched['AMCW']
        print(AMCW_img.shape,z_corr.shape)
        fname, _ = os.path.splitext(os.path.basename(data_path[0]))
        gen_batch = generator(cap_img)
        print(np.min(cap_img.numpy()),np.max(cap_img.numpy()))
        # plt.imshow(cap_img[0,0,:,:])
        # plt.show()
        show_gen_image(results_folder_path,fname, depth_img, gen_batch,z_corr, AMCW_img, i_batch)
        print(i_batch)