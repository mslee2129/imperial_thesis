import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import model
from torch.autograd import Variable

class CCGANModel(BaseModel):
    '''
    Implements a cGAN + one sided CycleGAN model
    '''

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm='instance', netG='unet_256', dataset_mode='aligned')
        return parser
    
    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.loss_names = ['G_A', 'G_B', 'D_A', 'D_B', 'cycle_A', 'cycle_B', 'idt_A', 'idt_B']
        self.visual_names_A = ['real_A', 'fake_B', 'rec_A']
        self.visual_names_B = ['real_B', 'fake_A', 'rec_B']

        if self.isTrain and self.opt.lambda_identity > 0.0:
            self.visual_names_A.append('idt_B')
            self.visual_names_B.append('idt_A')

        self.visual_names = self.visual_names_A + self.visual_names_B

        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B'] #', D_C', 'D_D']
        else:
            self.model_names = ['G_A', 'G_B']

        # define networks
        self.netG_A = model.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                     not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = model.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                     not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        
        if self.isTrain:
            self.netD_A = model.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D,
                                         opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = model.define_D(opt.input_nc, opt.ndf, opt.netD, opt.n_layers_D, 
                                         opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            # self.netD_C = model.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D,
            #                              opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            # self.netD_D = model.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D,
            #                              opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            
        if self.isTrain:
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = model.GANLoss(opt.gan_mode).to(self.device)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialise optimisers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_A = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_A.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_B = torch.optim.Adam(itertools.chain(self.netD_B.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            # self.optimizer_D_C = torch.optim.Adam(itertools.chain(self.netD_C.parameters(), self.netD_C.parameters()), lr=0.0001, betas=(opt.beta1, 0.999))
            # self.optimizer_D_D = torch.optim.Adam(itertools.chain(self.netD_D.parameters(), self.netD_D.parameters()), lr=0.0001, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D_A)
            self.optimizers.append(self.optimizer_D_B)
            # self.optimizers.append(self.optimizer_D_C)
            # self.optimizers.append(self.optimizer_D_D)

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.fake_B = self.netG_A(self.real_A) # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B) # G_B(B)
        self.rec_B = self.netG_A(self.fake_A) # G_A(G_B(B))

    def backward_D_basic(self, netD, real, fake):
        '''
        Calculate GAN loss for D
        '''
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D
    
    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    # def backward_D_C(self):
    #     real_A_and_real_B = torch.cat((self.real_A, self.real_B), 1)
    #     fake_A_and_real_B = torch.cat((self.fake_A, self.real_B), 1)

    #     real_A_and_real_B.requires_grad_()  # Ensure gradients are computed for this tensor
    #     fake_A_and_real_B.requires_grad_()  # Ensure gradients are computed for this tensor

    #     self.loss_D_C = self.backward_D_basic(self.netD_C, real_A_and_real_B, fake_A_and_real_B)

    #     # self.loss_D_C = self.backward_D_basic(self.netD_C, torch.cat((self.real_A,(self.real_B)),1), torch.cat((Variable(self.fake_A),(self.real_B)),1))

    # def backward_D_D(self):
    #     real_B_and_real_A = torch.cat((self.real_B, self.real_A), 1)
    #     fake_B_and_real_A = torch.cat((self.fake_B, self.real_A), 1)

    #     real_B_and_real_A.requires_grad_()  # Ensure gradients are computed for this tensor
    #     fake_B_and_real_A.requires_grad_()  # Ensure gradients are computed for this tensor

    #     self.loss_D_C = self.backward_D_basic(self.netD_C, real_B_and_real_A, fake_B_and_real_A)

        # self.loss_D_D = self.backward_D_basic(self.netD_D, torch.cat((self.real_B,(self.real_A)),1), torch.cat((Variable(self.fake_B),(self.real_A)),1))


    def backward_G(self):
        """Calculate the loss for generators"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0
       
        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A1 = self.criterionGAN(self.netD_D(torch.cat((self.fake_B, self.real_A),1)), True)#*lambda_A*lambda_idt
        self.loss_cycle_B1 = self.criterionGAN(self.netD_C(torch.cat((self.fake_A, self.real_B),1)), True)#*lambda_B*lambda_idt

        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_cycle_A1 + self.loss_cycle_B1 + self.loss_idt_A + self.loss_idt_B
        self.loss_G_B.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B, self.netD_C, self.netD_D], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # Discriminators
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        # D_A
        self.optimizer_D_A.zero_grad()
        self.backward_D_A()
        self.optimizer_D_A.step()
        # D_B
        self.optimizer_D_B.zero_grad()
        self.backward_D_B()
        self.optimizer_D_B.step()
        # # D_C
        # self.optimizer_D_C.zero_grad()
        # self.backward_D_C()
        # self.optimizer_D_C.step()
        # # D_D
        # self.optimizer_D_D.zero_grad()
        # self.backward_D_D()
        # self.optimizer_D_D.step() # update D_A and D_B's weights