import torch
import math
import random
import numpy
from data_utils import PCA_Handler

class SoundModel(torch.nn.Module):

    def __init__(self,mode='flat',n_z:int=250):

        super(SoundModel,self).__init__()
        self.model      = mode
        self.device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_z        = n_z
        self.mode       = mode
    
    def generate_rand(self,bs:int):
        if self.mode == 'flat':
            return torch.randn(size=(bs,self.n_z),device=self.device,dtype=torch.float)
        elif self.mode == 'conv':
            return torch.randn(size=(bs,1,self.n_z),device=self.device,dtype=torch.float)

class Generator(SoundModel):

    def __init__(self,n_z:int,n_out:int,handler:PCA_Handler,dropout:float=.1,act_fn=torch.nn.GELU):

        super(Generator,self).__init__(mode='flat')

        lin_act         = act_fn
        self.handler    = handler
        
        self.first      = torch.nn.Sequential(
            torch.nn.Linear(n_z,n_z*2),
            torch.nn.Dropout(dropout),
            lin_act(),

            torch.nn.Linear(n_z*2,n_z*2),
            torch.nn.Dropout(dropout),
            lin_act(),

            torch.nn.Linear(n_z*2,n_z*4),
            torch.nn.Dropout(dropout),
            lin_act(),

            torch.nn.Linear(n_z*4,n_z*4),
            torch.nn.Dropout(dropout),
            lin_act(),

            torch.nn.Linear(n_z*4,n_z*4),
            torch.nn.Dropout(dropout),
            lin_act(),

            torch.nn.Linear(n_z*4,n_z*4),
            torch.nn.Dropout(dropout),
            lin_act(),

            torch.nn.Linear(n_z*4,n_z*8),
            torch.nn.Dropout(dropout),
            lin_act(),

            torch.nn.Linear(n_z*8,n_out//4),
            torch.nn.Dropout(dropout),
            lin_act(),

            torch.nn.Linear(n_out//4,n_out//4),
            torch.nn.Dropout(dropout/2),
            lin_act(),

            torch.nn.Linear(n_out//4,n_out//2),
            torch.nn.Dropout(dropout/2),
            lin_act(),

            torch.nn.Linear(n_out//2,n_out//2),
            torch.nn.Dropout(dropout/2),
            lin_act(),

            torch.nn.Linear(n_out//2,n_out),
            torch.nn.Dropout(dropout/2),
            torch.nn.Tanh()
        )

        self.to(self.device)


    
    def forward(self,x:torch.Tensor)->torch.Tensor:
        x       = self.first(x)
        x       = self.handler.expand(x)
        return x



class Discriminator(SoundModel):


    def __init__(self,act_fn=torch.nn.GELU,dropout=.1):
        
        super(Discriminator, self).__init__(n_z=None)

        act_kwargs                      = {'in_place':True}
        bias                            = True


        self.conv_layers                = torch.nn.Sequential(
            torch.nn.Conv1d(1,16,63,2,63//2,bias=bias),                     # /2
            act_fn(**act_kwargs),

            torch.nn.Conv1d(16,64,5,2,5//2,bias=bias),                      # /4
            act_fn(**act_kwargs),

            torch.nn.Conv1d(64,128,7,2,7//2,bias=bias),                     # /8
            act_fn(**act_kwargs),

            torch.nn.Conv1d(128,128,7,2,7//2,bias=bias),                    # /16
            act_fn(**act_kwargs),

            torch.nn.Conv1d(128,256,7,2,7//2,bias=bias),                    # /32
            act_fn(**act_kwargs),

            torch.nn.Conv1d(256,256,7,2,7//2,bias=bias),                    # /64
            act_fn(**act_kwargs),

            torch.nn.Conv1d(256,512,7,2,7//2,bias=bias),                    # /128
            act_fn(**act_kwargs),

            torch.nn.Conv1d(512,512,7,2,7//2,bias=bias),                    # /256
            act_fn(**act_kwargs),

            torch.nn.Conv1d(512,1024,7,2,7//2,bias=bias),                   # /512
            act_fn(**act_kwargs),
            torch.nn.Flatten(1)
            
        )

        self.lin_layers                 = torch.nn.Sequential(
            torch.nn.Linear(1024,512),
            torch.nn.Dropout(dropout),

            torch.nn.Linear(512,128),
            torch.nn.Dropout(dropout),

            torch.nn.Linear(128,1),
            torch.nn.Sigmoid()
        )
              


        self.to(self.device)


    def forward(self, x:torch.Tensor)->torch.Tensor:
        x       = self.conv_layers(x)
        x       = self.lin_layers(x)
        return x 

