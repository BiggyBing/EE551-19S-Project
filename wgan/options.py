#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 22:28:00 2019

@author: bingyangwen
"""

class TrainingOptions:
    """
    Configuration options for the training
    """
    def __init__(self,
                 batch_size, number_of_epochs,
                 train_folder, validation_folder, runs_folder,
                 start_epoch, experiment_name):
        self.batch_size = batch_size
        self.number_of_epochs = number_of_epochs
        self.train_folder = train_folder
        self.runs_folder = runs_folder
        self.start_epoch = start_epoch
        self.experiment_name = experiment_name

class GanConfiguration():
    """
    The GAN network configuration.
    """
    def __init__(self, H: int, W: int, noise_length: int,
                 generate_num: int, critic_iters:int,
                 DIM: int):

        self.H = H
        self.W = W
        self.CRITIC_ITERS = critic_iters
        self.noise_length = noise_length
        # num_generation = generate_num ** 2
        self.generate_num = generate_num
        self.DIM = DIM
