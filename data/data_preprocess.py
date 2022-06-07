from __future__ import print_function, division
import torch
from torchvision import transforms
from torchvision.transforms import functional as TF
from matplotlib import pyplot as plt
from data import load

class ToTensor(object):
    def __call__(self, inputs):
        cap_image, depth_image = inputs['cap'].transpose((0, 2, 1)), inputs['depth_1024'].transpose((1, 0))
        return {'cap': torch.from_numpy(cap_image).to(torch.float32),
                'depth_1024': torch.unsqueeze(torch.from_numpy(depth_image).to(torch.float32), 0), 'path': inputs['path']}



class RandomCrop(object):
    def __call__(self, inputs):
        cap_image, depth_image = inputs['cap'], inputs['depth_1024']
        crop_size = transforms.RandomCrop.get_params(cap_image, (128, 128))
        return {'cap': TF.crop(cap_image, *crop_size),
                'depth_1024': TF.crop(depth_image, *crop_size), 'path': inputs['path']}


class ConstantCrop(object):
    def __call__(self, inputs):
        cap_image, depth_image = inputs['cap'], inputs['depth_1024']
        centercrop = transforms.CenterCrop((128, 128))
        crop_size = centercrop(cap_image)
        depth_image = centercrop(depth_image)
        return {'cap': crop_size,
                'depth_1024': depth_image, 'path': inputs['path']}





class RandomHorizontalFlip(object):
    def __call__(self, inputs):
        if torch.rand(1) < 0.5:
            cap_image, depth_image = inputs['cap'], inputs['depth_1024']
            return {'cap': TF.hflip(cap_image),
                    'depth_1024': TF.hflip(depth_image), 'path': inputs['path']}
        else:
            return inputs

class ImageNormalize(object):
    def __init__(self):
        self.ImgNorm = transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))
        self.ImgNorm_depth = transforms.Normalize((0.5), (0.5))
    def __call__(self, inputs):
        cap_image, depth_image = inputs['cap'], inputs['depth_1024']
        return {'cap': self.ImgNorm(cap_image),
                'depth_1024': self.ImgNorm_depth(depth_image), 'path': inputs['path']}



if __name__ == '__main__':

    # We can use an image folder dataset the way we have it setup.
    # Create the dataset
    dataset = load.ToFDataset(parent_path=r"C:\Users\919im\Documents\local-香川研\makeToFDataset\dataset_shizuoka",
                              transform=transforms.Compose([
                                  ToTensor(),
                                  RandomCrop(),
                                  RandomHorizontalFlip(),
                                  ImageNormalize(),
                              ]))
    revTrans = transforms.Normalize(mean=(-1), std=(2))

    for i in range(len(dataset)):
        inputs = dataset[i]
        cap_image, cap_ideal_image, depth_image = inputs['cap'], inputs['cap_ideal'],  inputs['depth_1024']
        path = inputs['path']
        print(i, cap_image.shape)
        print(i, cap_ideal_image.shape)
        print(i, depth_image.shape)

        fig = plt.figure()
        # ax.set_title('sample{}'.format(path))
        fig.add_subplot(3, 4, 1)
        plt.imshow((cap_image[0, :, :].numpy() + 1) / 2)
        fig.add_subplot(3, 4, 2)
        plt.imshow((cap_image[1, :, :].numpy() + 1) / 2)
        fig.add_subplot(3, 4, 3)
        plt.imshow((cap_image[2, :, :].numpy() + 1) / 2)
        fig.add_subplot(3, 4, 4)
        plt.imshow((cap_image[3, :, :].numpy() + 1) / 2)

        fig.add_subplot(3, 4, 5)
        plt.imshow((cap_ideal_image[0, :, :].numpy() + 1) / 2)
        fig.add_subplot(3, 4, 6)
        plt.imshow((cap_ideal_image[1, :, :].numpy() + 1) / 2)
        fig.add_subplot(3, 4, 7)
        plt.imshow((cap_ideal_image[2, :, :].numpy() + 1) / 2)
        fig.add_subplot(3, 4, 8)
        plt.imshow((cap_ideal_image[3, :, :].numpy() + 1) / 2)

        fig.add_subplot(3, 4, 9)
        plt.imshow((depth_image[0, :, :].numpy() + 1) / 2)
        fig.savefig('gen_image_{}.png'.format(1))
        plt.show()
        if i == 3:
            break

