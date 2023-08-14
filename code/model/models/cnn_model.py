import torch
from .base_model import BaseModel
from . import model

class CNNModel(BaseModel):
    '''
    '''

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser
    
    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.loss_names = ['L1']

        self.visual_names = ['real_A', 'fake_B', 'real_B']

        self.model_names = ['G']

        self.netG = model.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                   not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        
        if self.isTrain:
            self.criterionL1 = torch.nn.L1Loss()
            self.optimizer = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)

    def set_input(self, input):
        '''
        '''
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        '''
        '''
        self.fake_B = self.netG(self.real_A).to(self.device)

    def backward(self):
        self.loss_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        self.loss_L1.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()