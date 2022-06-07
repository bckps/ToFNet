from __future__ import print_function, division
import os
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from matplotlib import pyplot as plt
from models.model_cat_version import Generator, Discriminator, PatchDiscriminator
from data import data_preprocess_3dim
from models import total_variation
import csv
from data.load_npz_3dim import ToFDataset_3dim
from torch.optim.lr_scheduler import StepLR
from utils.compare_npphase import show_gen_image
from utils.weights_init import weights_init
from option.options import TrainOption
import shutil
from utils.four_phases_method import four_phases_depth,four_phases_cropped_depth

train_settings_file = 'option/train-settings.json'
opt = TrainOption(train_settings_file)
results_folder_path = os.path.join(opt.training_results_folder_name, opt.train_idname)
training_param_path = os.path.join(opt.training_param_folder_name, opt.train_idname)
os.makedirs(results_folder_path, exist_ok=True)
os.makedirs(training_param_path, exist_ok=True)
shutil.copyfile(train_settings_file, os.path.join(results_folder_path,'train-settings-{}.json'.format(opt.train_idname)))

# We can use an image folder dataset the way we have it setup.
# Create the dataset
dataset = ToFDataset_3dim(parent_path=opt.training_datasets_folder_name,
                          transform=transforms.Compose([
                              data_preprocess_3dim.ToTensor(),
                              data_preprocess_3dim.RandomCrop(),
                              # data_preprocess.ConstantCrop(),
                              data_preprocess_3dim.RandomHorizontalFlip(),
                              data_preprocess_3dim.ImageNormalize(),
                          ]))

dataset_val = ToFDataset_3dim(parent_path=opt.valid_datasets_folder_name,
                          transform=transforms.Compose([
                              data_preprocess_3dim.ToTensor(),
                              data_preprocess_3dim.RandomCrop(),
                              # data_preprocess.ConstantCrop(),
                              data_preprocess_3dim.RandomHorizontalFlip(),
                              data_preprocess_3dim.ImageNormalize(),
                          ]))

# Create the dataloader
dataloader = DataLoader(dataset, batch_size=opt.batch_size,
                        shuffle=True, num_workers=opt.workers)

val_loader = DataLoader(dataset_val, batch_size=opt.batch_size,
                        shuffle=True, num_workers=opt.workers)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and opt.ngpu > 0) else "cpu")

# Plot some training images
# img_batch = next(iter(dataloader))
# inp_batch, real_batch = img_batch['input'].to(device), img_batch['real'].to(device)


# Create the generator,discriminator
generator = Generator(in_nc=opt.INPUT_CHANNELS).to(device)
discriminator = PatchDiscriminator().to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (opt.ngpu > 1):
    generator = nn.DataParallel(generator, list(range(opt.ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.02.
generator.apply(weights_init)
discriminator.apply(weights_init)

# Initialize Loss functions
# adversarial_loss = nn.BCEWithLogitsLoss()
adversarial_loss = nn.MSELoss()
reconstruction_loss = nn.L1Loss()
totalvariation_loss = total_variation.TotalVariation()

# real_loss = nn.BCEWithLogitsLoss()
# generated_loss = nn.BCEWithLogitsLoss()
real_loss = nn.MSELoss()
generated_loss = nn.MSELoss()

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

#learning rate control
schedulerG = StepLR(optimizerG, step_size=opt.lr_decay_step, gamma=opt.lr_gammma, verbose=True)
schedulerD = StepLR(optimizerD, step_size=opt.lr_decay_step, gamma=opt.lr_gammma, verbose=True)

# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0
# torch.autograd.set_detect_anomaly(True)


# For each epoch
for epoch in range(opt.num_epochs):
    print(epoch)
    # For each batch in the dataloader
    for i, data_batch in enumerate(dataloader, 0):

        ############################
        # (1) Update G network
        ###########################
        # Format batch
        cap_img, depth_img = data_batch['cap'].to(device), data_batch['depth_1024'].to(device)


        gen_output = generator(cap_img)
        gen_output_tensor = gen_output.detach()
        disc_generated_output = discriminator(gen_output)

        adv_loss = opt.adv_weight * adversarial_loss(disc_generated_output, torch.ones_like(disc_generated_output))
        L1_loss = reconstruction_loss(gen_output, depth_img)
        tv_loss = opt.tv_weight * totalvariation_loss(gen_output,cap_img)
        gen_loss = adv_loss + L1_loss + tv_loss

        discriminator.zero_grad()
        generator.zero_grad()
        # Calculate gradients for G in backward pass
        gen_loss.backward()
        # gen_loss.backward(retain_graph=True)
        # Update G
        optimizerG.step()

        ############################
        # (2) Update D network
        ###########################
        disc_gen_output = discriminator(gen_output_tensor)
        disc_depth_output = discriminator(depth_img)

        D_real_loss = real_loss(disc_depth_output, torch.ones_like(disc_depth_output))
        D_gen_loss = generated_loss(disc_gen_output, torch.zeros_like(disc_gen_output))
        disc_loss =  D_real_loss + D_gen_loss

        discriminator.zero_grad()
        generator.zero_grad()
        # Calculate gradients for D
        disc_loss.backward()
        # Update D
        optimizerD.step()
        D_gen = disc_generated_output.mean().item()
        D_real = disc_depth_output.mean().item()

        # Output training stats
        if epoch % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(gen_x): %.4f\tD(real_x): %.4f '
                  % (epoch, opt.num_epochs, i, len(dataloader),
                     disc_loss.item(), gen_loss.item(), D_gen, D_real))
            vadv_loss_list = []
            vL1_loss_list = []
            vtv_loss_list = []
            vD_real_loss_list = []
            vD_gen_loss_list = []
            for v_i, val_data_batch in enumerate(val_loader, 0):
                vcap_img, vdepth_img = val_data_batch['cap'].to(device), val_data_batch['depth_1024'].to(device)
                with torch.no_grad():
                    vgen_images = generator(vcap_img)
                vz_corr = val_data_batch['phase'][0]
                vfname, _ = os.path.splitext(os.path.basename(val_data_batch['path'][0]))

                show_gen_image(results_folder_path, vfname, vdepth_img, vgen_images, vz_corr, epoch)

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': generator.state_dict(),
                    'optimizer_state_dict': optimizerG.state_dict(),
                    'loss': gen_loss,
                }, os.path.join(training_param_path, 'gen_{}times_model.pt'.format(epoch)))
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': discriminator.state_dict(),
                    'optimizer_state_dict': optimizerD.state_dict(),
                    'loss': disc_loss,
                }, os.path.join(training_param_path, 'disc_{}times_model.pt'.format(epoch)))

                vdisc_generated_output = discriminator(vgen_images)
                vadv_loss = opt.adv_weight * adversarial_loss(vdisc_generated_output,
                                                             torch.ones_like(vdisc_generated_output))
                vL1_loss = reconstruction_loss(vgen_images, vdepth_img)
                vtv_loss = opt.tv_weight * totalvariation_loss(vgen_images, vcap_img)
                vgen_loss = vadv_loss + vL1_loss + vtv_loss

                vgen_output_tensor = vgen_images.detach()
                vdisc_gen_output = discriminator(vgen_output_tensor)
                vdisc_depth_output = discriminator(vdepth_img)
                vD_real_loss = real_loss(vdisc_depth_output, torch.ones_like(vdisc_depth_output))
                vD_gen_loss = generated_loss(vdisc_gen_output, torch.zeros_like(vdisc_gen_output))
                vdisc_loss = D_real_loss + D_gen_loss

                vadv_loss_list.append(vadv_loss.item())
                vL1_loss_list.append(vL1_loss.item())
                vtv_loss_list.append(vtv_loss.item())
                vD_real_loss_list.append(vD_real_loss.item())
                vD_gen_loss_list.append(vD_gen_loss.item())
            # Save Losses for plotting later
            # G_losses.append(gen_loss.item())
            # D_losses.append(disc_loss.item())
            G_losses.append([epoch, np.mean(vadv_loss_list), np.mean(vL1_loss_list), np.mean(vtv_loss_list)])
            D_losses.append([epoch, np.mean(vD_real_loss_list), np.mean(vD_gen_loss_list)])


        # Check how the generator is doing by saving G's output on fixed_noise
        # if (iters % 1000 == 0) or ((epoch == opt.num_epochs - 1) and (i == len(dataloader) - 1)):
        #     with torch.no_grad():
        #         gen_images = generator(cap_img)
        #     z_corr = data_batch['phase'][0]
        #     fname, _ = os.path.splitext(os.path.basename(data_batch['path'][0]))
        #
        #     show_gen_image(results_folder_path, fname, depth_img, gen_images, z_corr, iters)
        #
        #     torch.save({
        #         'epoch': epoch,
        #         'model_state_dict': generator.state_dict(),
        #         'optimizer_state_dict': optimizerG.state_dict(),
        #         'loss': gen_loss,
        #     }, os.path.join(training_param_path, 'gen_{}times_model.pt'.format(iters)))
        #     torch.save({
        #         'epoch': epoch,
        #         'model_state_dict': discriminator.state_dict(),
        #         'optimizer_state_dict': optimizerD.state_dict(),
        #         'loss': disc_loss,
        #     }, os.path.join(training_param_path, 'disc_{}times_model.pt'.format(iters)))
        iters += 1
    schedulerG.step()
    schedulerD.step()

with open(os.path.join(results_folder_path, 'G-loss-log.csv'), 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['epoch', 'adv_loss', 'L1_loss', 'tv_loss'])
    writer.writerows(G_losses)

with open(os.path.join(results_folder_path, 'D-loss-log.csv'), 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['epoch', 'D_real_loss', 'D_gen_loss'])
    writer.writerows(D_losses)