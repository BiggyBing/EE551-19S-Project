#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 15:40:37 2019

@author: bingyangwen
"""

import os
import time
import pprint
import argparse
import torch
import numpy as np
import pickle
import utils
import logging
import sys

from options import *
from model.wgan import WGAN




def train(model: WGAN,
          device: torch.device,
          gan_config: GanConfiguration,
          train_options: TrainingOptions,
          this_run_folder: str):
    """
    Trains the HiDDeN model
    :param model: The model
    :param device: torch.device object, usually this is GPU (if avaliable), otherwise CPU.
    :param gan_config: The network configuration
    :param train_options: The training settings
    :param this_run_folder: The parent folder for the current training run to store training artifacts/results/logs.
    
    :return:
    """

    train_data = utils.get_cifar_loaders(gan_config, train_options)
    file_count = len(train_data.dataset)
    if file_count % train_options.batch_size == 0:
        steps_in_epoch = file_count // train_options.batch_size
    else:
        steps_in_epoch = file_count // train_options.batch_size + 1

    print_each = 10
    #generating images per gen_epoch cycles
    gen_epoch = 10
    images_to_save = 8
    saved_images_size = (32, 32)

    for epoch in range(train_options.start_epoch, train_options.number_of_epochs + 1):
        logging.info('\nStarting epoch {}/{}'.format(epoch, train_options.number_of_epochs))
        logging.info('Batch size = {}\nSteps in epoch = {}'.format(train_options.batch_size, steps_in_epoch))
        losses_accu = {}
        epoch_start = time.time()
        step = 1
            
        for image, _ in train_data:
            image = image.to(device)
            noise = torch.Tensor(np.random.normal(size=(image.shape[0],gan_config.noise_length))).to(device)
            losses = model.train_on_batch([image, noise])
            if not losses_accu: # dict is empty, initialize
                for name in losses:
                    losses_accu[name] = []

            for name, loss in losses.items():
                losses_accu[name].append(loss)
            if step % print_each == 0 or step == steps_in_epoch:
                logging.info('Epoch: {}/{} Step: {}/{}'.format(epoch, train_options.number_of_epochs, step, steps_in_epoch))
                utils.print_progress(losses_accu)
                logging.info('-' * 40)
            step += 1

        train_duration = time.time() - epoch_start
        logging.info('Epoch {} training duration {:.2f} sec'.format(epoch, train_duration))
        logging.info('-' * 40)
        utils.write_losses(os.path.join(this_run_folder, 'train.csv'), losses_accu, epoch, train_duration)
        
        
        if epoch % gen_epoch == 0:
            logging.info('Generating images for epoch {}/{}'.format(epoch, train_options.number_of_epochs))
            gen_noise = torch.Tensor(np.random.normal(size=(gan_config.generate_num**2, gan_config.noise_length)))
            fake_images = model.generate_images(gen_noise.to(device))
            
            utils.save_images(fake_images.cpu(),
                              gan_config.generate_num,
                              epoch,
                              os.path.join(this_run_folder, 'images'), resize_to=saved_images_size)

        utils.print_progress(losses_accu)
        logging.info('-' * 40)
        utils.save_checkpoint(model, train_options.experiment_name, epoch, os.path.join(this_run_folder, 'checkpoints'))
        
        
def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    parser = argparse.ArgumentParser(description='Training of WGAN nets')
    parser.add_argument('--data-dir', '-d', default = '', type=str, help='The directory where the data is stored.')
    parser.add_argument('--batch-size', '-b', default = 100, type=int, help='The batch size.')
    #parser.add_argument('--LAMBDA', '-l', default = 10, type=int, help='Gradient penalty lambda hyperparameter.')
    parser.add_argument('--epochs', '-e', default=30,type=int, help='Number of epochs to run the simulation.')
    parser.add_argument('--name', default = 'WGAN_Cifar10_first_test', type=str, help='The name of the experiment.')
    parser.add_argument('--filter', '-f', default=64, type=int, help='Parameter to control the filter numbers in network.')
    parser.add_argument('--runs-folder', '-sf', default=os.path.join('.', 'runs'), type=str, help='The root folder where data about experiments are stored.')
    parser.add_argument('--size', '-s', default=32, type=int, help='The size of the images (images are square so this is height and width).')
    parser.add_argument('--noise', '-m', default=128, type=int, help='The length in random sample noise')
    parser.add_argument('--continue-from-folder', '-c', default='', type=str, help='The folder from where to continue a previous run. Leave blank if you are starting a new experiment.')
    

    
    parser.set_defaults()
    args = parser.parse_args()

    checkpoint = None
    if args.continue_from_folder != '':
        this_run_folder = args.continue_from_folder
        options_file = os.path.join(this_run_folder, 'options-and-config.pickle')
        train_options, gan_config = utils.load_options(options_file)
        checkpoint = utils.load_last_checkpoint(os.path.join(this_run_folder, 'checkpoints'))
        train_options.start_epoch = checkpoint['epoch']+1
    else:
        start_epoch = 1
        train_options = TrainingOptions(
            batch_size=args.batch_size,
            number_of_epochs=args.epochs,
            train_folder=os.path.join(args.data_dir, 'train'),
            validation_folder=os.path.join(args.data_dir, 'val'),
            runs_folder=os.path.join('.', 'runs'),
            start_epoch=start_epoch,
            experiment_name=args.name)

        gan_config = GanConfiguration(H=args.size, W=args.size,
                                      noise_length=args.noise, DIM = args.filter,
                                      generate_num=10, critic_iters = 5)

        this_run_folder = utils.create_folder_for_run(train_options.runs_folder, args.name)
        with open(os.path.join(this_run_folder, 'options-and-config.pickle'), 'wb+') as f:
            pickle.dump(train_options, f)
            pickle.dump(gan_config, f)

    logging.basicConfig(level=logging.INFO,
                        format='%(message)s',
                        handlers=[
                            logging.FileHandler(os.path.join(this_run_folder, args.name+'log')),
                            logging.StreamHandler(sys.stdout)
                        ])

    model = WGAN(gan_config, device)

    if args.continue_from_folder != '':
        # if we are continuing, we have to load the model params
        assert checkpoint is not None
        utils.model_from_checkpoint(model, checkpoint)

    logging.info('WGAN model: {}\n'.format(model.to_stirng()))
    logging.info('Model Configuration:\n')
    logging.info(pprint.pformat(vars(gan_config)))
    logging.info('\nTraining train_options:\n')
    logging.info(pprint.pformat(vars(train_options)))


    train(model, device, gan_config, train_options, this_run_folder)
    
if __name__ == '__main__':
    main()

