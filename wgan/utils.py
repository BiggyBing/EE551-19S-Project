#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 15:56:13 2019

@author: bingyangwen
"""

import numpy as np
import os
import re
import csv
import time
import pickle
import logging

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import torchvision.utils
from torch.utils import data
import torch.nn.functional as F

from options import GanConfiguration, TrainingOptions
from model.wgan import WGAN

class NumpyDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, file_path, transform=None):
        """
        Args:
            file_pat (string): Path to the numpy file .
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        data = np.load(file_path)
        data = torch.from_numpy(data)
        if len(data.shape) == 3:
            data.unsqueeze(0)
        data = data.permute(0,3,1,2)
        self.data = data.numpy()
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        return sample
    
def image_to_tensor(image):
    """
    Transforms a numpy-image into torch tensor
    :param image: (batch_size x height x width x channels) uint8 array
    :return: (batch_size x channels x height x width) torch tensor in range [-1.0, 1.0]
    """
    image_tensor = torch.Tensor(image)
    image_tensor.unsqueeze_(0)
    image_tensor = image_tensor.permute(0, 3, 1, 2)
    image_tensor = image_tensor / 127.5 - 1
    return image_tensor


def tensor_to_image(tensor):
    """
    Transforms a torch tensor into numpy uint8 array (image)
    :param tensor: (batch_size x channels x height x width) torch tensor in range [-1.0, 1.0]
    :return: (batch_size x height x width x channels) uint8 array
    """
    image = tensor.permute(0, 2, 3, 1).cpu().numpy()
    image = (image + 1) * 127.5
    return np.clip(image, 0, 255).astype(np.uint8)


def save_images(original_images, num_gen, epoch, folder, resize_to=None):

    images = original_images[:original_images.shape[0], :, :, :].cpu()

    # scale values to range [0, 1] from original range of [-1, 1]
    images = (images + 1) / 2
    
    if resize_to is not None:
        images = F.interpolate(images, size=resize_to)
       
     
    filename = os.path.join(folder, 'epoch-{}.png'.format(epoch))
    torchvision.utils.save_image(images, filename, num_gen, normalize=False)


def sorted_nicely(l):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def last_checkpoint_from_folder(folder: str):
    last_file = sorted_nicely(os.listdir(folder))[-1]
    last_file = os.path.join(folder, last_file)
    return last_file


def save_checkpoint(model: WGAN, experiment_name: str, epoch: int, checkpoint_folder: str):
    """ Saves a checkpoint at the end of an epoch. """
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    checkpoint_filename = experiment_name+'--epoch-{epoch}.pyt'
    checkpoint_filename = os.path.join(checkpoint_folder, checkpoint_filename)
    logging.info('Saving checkpoint to {}'.format(checkpoint_filename))
    checkpoint = {
        'gen-model': model.generator.state_dict(),
        'gen-optim': model.gen_optimizer.state_dict(),
        'disc-model': model.discriminator.state_dict(),
        'disc-optim': model.disc_optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, checkpoint_filename)
    logging.info('Saving checkpoint done.')


# def load_checkpoint(hidden_net: Hidden, options: Options, this_run_folder: str):
def load_last_checkpoint(checkpoint_folder):
    """ Load the last checkpoint from the given folder """
    last_checkpoint_file = last_checkpoint_from_folder(checkpoint_folder)
    logging.info("=> loading checkpoint '{}'".format(last_checkpoint_file))
    checkpoint = torch.load(last_checkpoint_file)
    logging.info("=> loaded checkpoint '{}' (epoch {})".format(last_checkpoint_file, checkpoint['epoch']))

    return checkpoint


def model_from_checkpoint(wgan_net, checkpoint):
    """ Restores the hidden_net object from a checkpoint object """
    wgan_net.generator.load_state_dict(checkpoint['gen-model'])
    wgan_net.gen_optimizer.load_state_dict(checkpoint['gen-optim'])
    wgan_net.discriminator.load_state_dict(checkpoint['disc-model'])
    wgan_net.disc_optimizer.load_state_dict(checkpoint['disc-optim'])


def load_options(options_file_name) -> (TrainingOptions, GanConfiguration, dict):
    """ Loads the training, model, and noise configurations from the given folder """
    with open(os.path.join(options_file_name), 'rb') as f:
        train_options = pickle.load(f)
        gan_config = pickle.load(f)

    return train_options, gan_config


def get_data_loaders(gan_config: GanConfiguration, train_options: TrainingOptions):
    """ Get torch data loaders for training and validation. The data loaders take a crop of the image,
    transform it into tensor, and normalize it."""
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomCrop((gan_config.H, gan_config.W), pad_if_needed=True),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'test': transforms.Compose([
            transforms.CenterCrop((gan_config.H, gan_config.W)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }

    train_images = datasets.ImageFolder(train_options.train_folder, data_transforms['train'])
    train_loader = torch.utils.data.DataLoader(train_images, batch_size=train_options.batch_size, shuffle=True, num_workers=4)

    validation_images = datasets.ImageFolder(train_options.validation_folder, data_transforms['test'])
    validation_loader = torch.utils.data.DataLoader(validation_images, batch_size=train_options.batch_size, shuffle=False, num_workers=4)

    return train_loader, validation_loader


def get_cifar_loaders(gan_config: GanConfiguration, train_options: TrainingOptions):
    """ Get torch data loaders for training and validation. The data loaders take a crop of the image,
    transform it into tensor, and normalize it."""
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomCrop((gan_config.H, gan_config.W), pad_if_needed=True),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'test': transforms.Compose([
            transforms.CenterCrop((gan_config.H, gan_config.W)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }
    data_path = os.path.join(os.getcwd(),'data/cifar')

    download = False if os.path.isdir(data_path) else True

    train_images = datasets.CIFAR10(data_path, train=True, transform=data_transforms['train'],download=download)
    train_loader = torch.utils.data.DataLoader(train_images, batch_size=train_options.batch_size, shuffle=True, num_workers=4)


    return train_loader


def get_wm_cifar_loaders(file_name, device, gan_config: GanConfiguration, train_options: TrainingOptions):

    data_transforms = transforms.Compose([
            transforms.RandomCrop((gan_config.H, gan_config.W), pad_if_needed=True),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    
    data_path = os.path.join(os.getcwd()+'/watermarked_data/', file_name)
    
    images = NumpyDataset(data_path, transform = data_transforms)

    image_loader = torch.utils.data.DataLoader(images, batch_size=train_options.batch_size, shuffle=True, num_workers=4)

    return image_loader


def print_progress(losses_accu):
    max_len = max([len(loss_name) for loss_name in losses_accu])
    for loss_name, loss_value in losses_accu.items():
        logging.info(loss_name.ljust(max_len+4) + '{:.4f}'.format(np.mean(loss_value)))


def create_folder_for_run(runs_folder, experiment_name):

    if not os.path.exists(runs_folder):
        os.makedirs(runs_folder)

    this_run_folder = os.path.join(runs_folder, experiment_name+time.strftime("%Y.%m.%d--%H-%M-%S"))

    os.makedirs(this_run_folder)
    os.makedirs(os.path.join(this_run_folder, 'checkpoints'))
    os.makedirs(os.path.join(this_run_folder, 'images'))

    return this_run_folder


def write_losses(file_name, losses_accu, epoch, duration):
    with open(file_name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if epoch == 1:
            row_to_write = ['epoch'] + [loss_name.strip() for loss_name in losses_accu.keys()] + ['duration']
            writer.writerow(row_to_write)
        row_to_write = [epoch] + ['{:.4f}'.format(np.mean(loss_list)) for loss_list in losses_accu.values()] + ['{:.0f}'.format(duration)]
        writer.writerow(row_to_write)