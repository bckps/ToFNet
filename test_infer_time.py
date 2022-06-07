from __future__ import print_function, division
import os, time
import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils.four_phases_method import four_phases_cropped_depth
from utils.compare_npphase import show_gen_image
from data.load_npz_3dim import ToFDataset_3dim
from data import data_preprocess_3dim
from models.model_cat_version import Generator

batch_size = 1
# Each image is 256x256 in size
IMG_WIDTH = 256
IMG_HEIGHT = 256
INPUT_CHANNELS = 3
OUTPUT_CHANNELS = 1
LAMBDA = 100


if __name__ == '__main__':

    test_dataset_folder = r'datasets/v3-corner-3rd'
    trained_model_path = r'checkpoints/appendix3-flip-corner-v3-3rd-adv015-tv8e-7-beta05/gen_1500times_model.pt'

    infer_time_list = []
    gen_MAE_list = []
    phase_MAE_list = []
    gen_var_list = []
    phase_var_list = []

    dataset = ToFDataset_3dim(parent_path=test_dataset_folder,
                              transform=transforms.Compose([
                                  data_preprocess_3dim.ToTensor(),
                                  data_preprocess_3dim.ConstantCrop(),
                                  data_preprocess_3dim.ImageNormalize(),
                              ]))
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=0,worker_init_fn = lambda id: np.random.seed(42))

    generator = Generator(in_nc=INPUT_CHANNELS)
    generator.load_state_dict(torch.load(trained_model_path)['model_state_dict'])
    generator.eval()



    for i_batch, sample_batched in enumerate(dataloader):
        cap_img, depth_img, z_corr, data_path = sample_batched['cap'],  \
                                            sample_batched['depth_1024'], sample_batched['phase'][0], sample_batched['path']

        fname, _ = os.path.splitext(os.path.basename(data_path[0]))
        start = time.time()
        gen_batch = generator(cap_img)
        infer_time = time.time() - start
        infer_time_list.append(infer_time)

        depth_img = 6.6 * (depth_img[0, 0, :, :].detach().cpu().numpy() + 1) / 2
        gen_img = 6.6 * (gen_batch[0, 0, :, :].detach().cpu().numpy() + 1) / 2
        z_corr = z_corr[0, :, :].detach().cpu().numpy()

        offset = 0
        phase_MAE = np.mean(np.abs(z_corr-depth_img))
        gen_MAE = np.mean(np.abs(gen_img - depth_img))

        phase_var = np.sqrt(np.mean(np.square(np.abs(z_corr - depth_img))))
        gen_var = np.sqrt(np.mean(np.square(np.abs(gen_img - depth_img))))

        phase_MAE_list.append(phase_MAE)
        gen_MAE_list.append(gen_MAE)

        phase_var_list.append(phase_var)
        gen_var_list.append(gen_var)
        print(i_batch)

    print(f'phase variance: {np.mean(phase_var_list)}')
    print(f'gen variance: {np.mean(gen_var_list)}')
    print(f'phase MAE: {np.mean(phase_MAE_list)}')
    print(f'gen MAE: {np.mean(gen_MAE_list)}')
    print(f'infer time average: {np.mean(infer_time_list)}')