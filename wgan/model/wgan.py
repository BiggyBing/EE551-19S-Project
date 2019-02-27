#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 13:40:06 2019

@author: bingyangwen
"""

import numpy as np
import torch
from torch import nn
from torch import autograd
from torch import optim
from options import GanConfiguration
from model.generator import Generator
from model.discriminator import Discriminator

class WGAN:
    def __init__(self, configuration: GanConfiguration, device: torch.device):
        """
        :param configuration: Configuration for the net, such as the size of the input image, number of channels in the intermediate layers, etc.
        :param device: torch.device object, CPU or GPU
        """
        super(WGAN, self).__init__()

        self.generator = Generator(configuration).to(device)
        self.gen_optimizer = torch.optim.Adam(self.generator.parameters())

        self.discriminator = Discriminator(configuration).to(device)
        self.disc_optimizer = torch.optim.Adam(self.discriminator.parameters())
        
        ########### Inception Score ################

        self.config = configuration
        self.device = device

    def calc_gradient_penalty(self, real_data, fake_data):
        LAMBDA = 10  # Gradient penalty lambda hyperparameter
        BATCH_SIZE = real_data.shape[0]
        alpha = torch.rand(BATCH_SIZE, 1)
        alpha = alpha.expand(BATCH_SIZE, real_data.nelement() // BATCH_SIZE).contiguous().view(BATCH_SIZE, 3, 32, 32)
        alpha = alpha.to(self.device)

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = autograd.Variable(interpolates, requires_grad=True)
        disc_interpolates = self.discriminator(interpolates)

        grad_outputs = (torch.ones(disc_interpolates.size())).to(self.device)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=grad_outputs,
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA

        return gradient_penalty

    def train_on_batch(self, batch: list):
        """
        Trains the network on a single batch 
        :param batch: batch of training data, in the form [images, messages]
        :return: 
            losses: dictionary of error metrics from Generator and Discriminator on the current batch
            fake_images: generated images 
        """
        real_images, noise = batch
        batch_size = real_images.shape[0]
        self.generator.train()
        self.discriminator.train()

        # ####
        # one = torch.
        # mone = one * -1
        # one = one.to(self.device)
        # mone = mone.to(self.device)
        # ####
        #
        for var in self.discriminator.parameters():
            var.requires_grad = True      # Set False when train generator
            
        for i in range(self.config.CRITIC_ITERS):
            self.discriminator.zero_grad()
            
            D_real = self.discriminator(real_images)
            D_real = D_real.mean()
            
            fake_images = self.generator(noise)
            D_fake = self.discriminator(fake_images)
            D_fake = D_fake.mean()
            
            #train with gradient penalty
            gradient_penalty = self.calc_gradient_penalty(real_images, fake_images)

            D_cost = D_fake - D_real + gradient_penalty

            D_cost.backward()
            Wasserstein_D = D_real - D_fake

            self.disc_optimizer.step()
        
        for var in self.discriminator.parameters():
            var.requires_grad = False # Avoid redundant calculation
        
        self.generator.zero_grad()
        
        fake_images = self.generator(noise)
        G = self.discriminator(fake_images)
        G = G.mean()
        G_cost = -G
        G_cost.backward()
        self.gen_optimizer.step()
        
        losses = {
            'G_cost         ': G_cost.data.cpu().numpy(),
            'D_cost         ': D_cost.data.cpu().numpy(),
            'Wasserstein_D  ': Wasserstein_D.data.cpu().numpy(),
        }
        return losses
    
    def generate_images(self, noise):
        
        fake_images = self.generator(noise)
        
        return fake_images
    
    def to_stirng(self):
        
        return '{}\n{}'.format(str(self.generator), str(self.discriminator))
        
        



