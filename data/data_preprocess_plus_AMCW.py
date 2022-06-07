from __future__ import print_function, division
import torch
from torchvision import transforms
from torchvision.transforms import functional as TF
from matplotlib import pyplot as plt
from data.load_npz_3dim import ToFDataset_3dim

class ToTensor(object):
    def __call__(self, inputs):
        return {'cap': torch.from_numpy(inputs['cap']).to(torch.float32),
                'depth_1024': torch.unsqueeze(torch.from_numpy(inputs['depth_1024']).to(torch.float32), 0),
                'phase': torch.unsqueeze(torch.from_numpy(inputs['phase']), 0),
                'AMCW': torch.unsqueeze(torch.from_numpy(inputs['AMCW']), 0), 'path': inputs['path']}


class RandomCrop(object):
    def __call__(self, inputs):
        cap_image, depth_image, phase_image = inputs['cap'], inputs['depth_1024'], inputs['phase']
        crop_size = transforms.RandomCrop.get_params(cap_image, (128, 128))
        return {'cap': TF.crop(cap_image, *crop_size),
                'depth_1024': TF.crop(depth_image, *crop_size),
                'phase': TF.crop(phase_image, *crop_size), 'path': inputs['path']}


class ConstantCrop(object):
    def __call__(self, inputs):
        cap_image, depth_image, phase_image, AMCW_image = inputs['cap'], inputs['depth_1024'], inputs['phase'], inputs['AMCW']
        centercrop = transforms.CenterCrop((128, 128))
        crop_size = centercrop(cap_image)
        depth_image = centercrop(depth_image)
        phase_image = centercrop(phase_image)
        AMCW_image = centercrop(AMCW_image)
        return {'cap': crop_size,
                'depth_1024': depth_image,
                'phase': phase_image,
                'AMCW': AMCW_image, 'path': inputs['path']}

class panaToFCrop(object):
    def __call__(self, inputs):
        cap_image, depth_image, phase_image = inputs['cap'], inputs['depth_1024'], inputs['phase']
        return {'cap': TF.crop(cap_image, 250, 300, 128, 128),
                'depth_1024': TF.crop(depth_image, 250, 300, 128, 128),
                'phase': TF.crop(phase_image, 250, 300, 128, 128), 'path': inputs['path']}

class RandomHorizontalFlip(object):
    def __call__(self, inputs):
        if torch.rand(1) < 0.5:
            cap_image, depth_image, phase_image = inputs['cap'], inputs['depth_1024'], inputs['phase']
            return {'cap': TF.hflip(cap_image),
                    'depth_1024': TF.hflip(depth_image),
                    'phase': TF.hflip(phase_image),'path': inputs['path']}
        else:
            return inputs

class ImageNormalize(object):
    def __init__(self):
        self.depth_scale = 6.6
        # self.sensescale = 100.19024438899942
        # self.ImgNorm = transforms.Normalize((3.11842938, 6.24030427, 0.68059899), (22.14534125, 5.73035384, 0.21797184))
        # self.ImgNorm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.ImgNorm_depth = transforms.Normalize((0.5), (0.5))
    def __call__(self, inputs):
        cap_image, depth_image = inputs['cap'], inputs['depth_1024']
        return {'cap': cap_image,#'cap': self.ImgNorm(cap_image/self.sensescale),
                'depth_1024': self.ImgNorm_depth(depth_image/self.depth_scale),
                'phase': inputs['phase'],
                'AMCW': inputs['AMCW'],'path': inputs['path']}


class _ImageNormalize_debug(object):
    def __init__(self):
        self.depth_scale = 6.6
        self.sensescale = 100.19024438899942
        # self.ImgNorm = transforms.Normalize((3.11842938, 6.24030427, 0.68059899), (22.14534125, 5.73035384, 0.21797184))
        # self.ImgNorm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.ImgNorm_depth = transforms.Normalize((0.5), (0.5))
    def __call__(self, inputs):
        cap_image, depth_image = inputs['cap'], inputs['depth_1024']
        return {'cap': cap_image/self.sensescale,#'cap': self.ImgNorm(cap_image/self.sensescale),
                'depth_1024': self.ImgNorm_depth(depth_image/self.depth_scale),
                # 'origin_1024': depth_image,
                'phase': inputs['phase'], 'path': inputs['path']}


if __name__ == '__main__':

    # We can use an image folder dataset the way we have it setup.
    # Create the dataset
    dataset = ToFDataset_3dim(parent_path=r"/home/saijo/labwork/pytorchDToF-edit-panaToF/datasets/train_3dim",
                              transform=transforms.Compose([
                                  ToTensor(),
                                  RandomCrop(),
                                  RandomHorizontalFlip(),
                                  # ImageNormalize(),
                                  _ImageNormalize_debug(),
                              ]))


    for i in range(len(dataset)):
        inputs = dataset[i]
        debug = False
        if debug:
            cap_image, depth_image, depth_origin, phase_image = inputs['cap'], inputs['depth_1024'], inputs['origin_1024'], inputs['phase']
        else:
            cap_image, depth_image, phase_image = inputs['cap'], inputs['depth_1024'], inputs['phase']

        path = inputs['path']
        print(i, cap_image.shape)
        print(i, phase_image.shape)
        print(i, depth_image.shape)

        fig = plt.figure()
        plt.imshow(phase_image[0, :, :].numpy())
        plt.colorbar()
        # fig.savefig('gen_image_{}.png'.format(1))
        plt.show()


        fig = plt.figure()
        plt.imshow(6.6 * (depth_image[0, :, :].numpy() + 1) / 2)
        plt.colorbar()
        # fig.savefig('gen_image_{}.png'.format(1))
        plt.show()

        if debug:
            fig = plt.figure()
            plt.imshow(depth_origin[0, :, :].numpy())
            plt.colorbar()
            # fig.savefig('gen_image_{}.png'.format(1))
            plt.show()

        fig = plt.figure()
        plt.imshow(depth_image[0, :, :].numpy())
        plt.colorbar()
        # fig.savefig('gen_image_{}.png'.format(1))
        plt.show()
        if i == 3:
            break

