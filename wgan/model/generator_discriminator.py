#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn
from model.encoder import Generator
from model.decoder import Discriminator
from options import GanConfiguration



class GeneratorDiscriminator(nn.Module):
    """
    Combines generator -> discriminator into single pipeline.
    The input is the random noise and images. 
    """
    def __init__(self, config: HiDDenConfiguration, noiser: Noiser):

        super(GeneratorDiscriminator, self).__init__()
        self.generator = Generator(config)

        self.discriminator = Discriminator(config)

    def forward(self, noise, image):
        real = image
        fake = self.encoder(image, message)
        
        real_logit = self.discriminator(real)
        fake_logit = self.discriminator(fake)
        
        return real_logit, fake_logit

