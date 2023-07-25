import torch
import itertools
from util.image_pool import ImagePool
from base_model import BaseModel
import model

class GANModel(BaseModel):
    '''
    Implements a cGAN + one sided CycleGAN model
    '''

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return super().modify_commandline_options(parser, is_train)
    
    def __init__(self, opt):
        super().__init__(self, opt)

        self.loss_names = ['G_A', 'G_B', 'D_A', 'D_B', 'cycle_A']
        self.visual_names_A = ['real_A', 'fake_B', 'real_B']
        self.visual_names_B = ['real_B', 'fake_A', 'rec_B']

        if self.isTrain and self.opt.lambda_identity > 0.0:
            self.visual_names_A.append('idt_B')
            self.visual_names_B.append('idt_A')

        self.visual_names = self.visual_names_A + self.visual_names_B

        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:
            self.model_names = ['G_A', 'G_B']

        # define networks
        self.netG_A = model.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                     not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = model.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                     not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        
        if self.isTrain:
            self.netD_A = model.define_D(opt.output_nc, opt.ndf, opt.netD, opt.norm
                                         not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = model.define_D(opt.output_nc, opt.ndf, opt.netD, opt.norm
                                         not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)