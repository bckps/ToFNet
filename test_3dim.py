from __future__ import print_function, division
import os
import torch
import numpy as np
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


if __name__ == '__main__':


    test_dataset_folder = r"datasets/test_rev2"
    eval_folder = 'eval'
    nameprefix = 'test-rev2-adv01-tv0-beta07'
    trained_model_path = r'checkpoints/train-rev2-adv01-tv0-beta07/gen_17999times_model.pt'


    dataset = ToFDataset_3dim(parent_path=test_dataset_folder,
                              transform=transforms.Compose([
                                  data_preprocess_3dim.ToTensor(),
                                  data_preprocess_3dim.ConstantCrop(),
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

        show_gen_image(results_folder_path,fname, depth_img, gen_batch,z_corr, i_batch)
        print(i_batch)