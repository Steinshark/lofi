import torch
import math
import random
import numpy
from data_utils import PCA_Handler
import torchaudio 

class SoundModel(torch.nn.Module):

    def __init__(self,mode='flat',n_z:int=250):

        super(SoundModel,self).__init__()
        self.mode      = mode
        self.device     = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.n_z        = n_z
    
    def generate_rand(self,bs:int):
        if self.mode == 'flat':
            return torch.randn(size=(bs,self.n_z),device=self.device,dtype=torch.float)
        elif self.mode == 'conv':
            return torch.randn(size=(bs,self.n_z,1),device=self.device,dtype=torch.float)
    
    def size(self):
        return sum([p.numel()*p.element_size() for p in self.parameters()])
    
    def params(self):
        return sum([p.numel()for p in self.parameters()])



class LinearResBlock(torch.nn.Module):

    def __init__(self,n_neurons_in,n_neurons_out,act_fn=torch.nn.Tanh,dropout_p=.1,residual=True):
            
        super(LinearResBlock,self).__init__()
        self.residual   = residual
        self.block  = torch.nn.Sequential(
            torch.nn.Linear(n_neurons_in,n_neurons_out),
            torch.nn.Dropout(p=dropout_p),
            act_fn()
        )
    
    def forward(self,x:torch.Tensor)->torch.Tensor:
        if self.residual:
            return self.block(x) + x 
        else:
            return self.block(x)
        


class ResGenerator(SoundModel):

    def __init__(self,n_z:int,n_out:int,handler:PCA_Handler,dropout:float=.1,act_fn=torch.nn.GELU,n_block:int=8):

        super(ResGenerator,self).__init__(mode='flat',n_z=n_z)

        lin_act         = act_fn
        self.handler    = handler   
        rep_dim         = 4096



        self.adapter1   = LinearResBlock(n_z,rep_dim//4,residual=False,act_fn=act_fn,dropout_p=.5)
        self.block1     = torch.nn.Sequential()

        self.adapter2   = LinearResBlock(rep_dim//4,rep_dim//4,residual=False,act_fn=act_fn,dropout_p=.5)
        self.block2     = torch.nn.Sequential()

        self.adapter3   = LinearResBlock(rep_dim//4,rep_dim//2,residual=False,act_fn=act_fn,dropout_p=.5)
        self.block3     = torch.nn.Sequential()

        self.adapter4   = LinearResBlock(rep_dim//2,rep_dim,residual=False,act_fn=act_fn,dropout_p=.5)
        self.block4     = torch.nn.Sequential()


        for i in range(n_block-1):
            self.block1.append(LinearResBlock(rep_dim//4,rep_dim//4,act_fn=act_fn,dropout_p=.5))
            self.block2.append(LinearResBlock(rep_dim//4,rep_dim//4,act_fn=act_fn,dropout_p=.5))
            self.block3.append(LinearResBlock(rep_dim//2,rep_dim//2,act_fn=act_fn,dropout_p=.5))
            self.block4.append(LinearResBlock(rep_dim,rep_dim,act_fn=act_fn,dropout_p=.5))
        

        self.out        = torch.nn.Sequential(
            torch.nn.Linear(rep_dim,n_out),
            torch.nn.Tanh()
        )
        

        self.expander = self.handler.constructor.clone().to(self.device)
        self.to(self.device)


    
    def forward(self,x:torch.Tensor)->torch.Tensor:
        x       = self.adapter1(x)
        x       = self.block1(x)
        x       = self.adapter2(x)
        x       = self.block2(x)
        x       = self.adapter3(x)
        x       = self.block3(x)
        x       = self.adapter4(x)
        x       = self.block4(x)
        x       = self.out(x)
        x       = x.T
        x       = torch.cat([self.expander @ x]).T
        return x



class Generator2(SoundModel):

    def __init__(self,n_z:int,n_out:int,handler:PCA_Handler,dropout:float=.1,act_fn=torch.nn.GELU,bs:int=8):

        super(Generator2,self).__init__(mode='flat',n_z=n_z)

        lin_act         = act_fn
        self.handler    = handler
        
        self.block1      = torch.nn.Sequential(
            torch.nn.Linear(n_z,n_z),
            torch.nn.Dropout(dropout),
            lin_act()
        )

        self.block2      = torch.nn.Sequential(
            torch.nn.Linear(n_z,n_z),
            torch.nn.Dropout(dropout),
            lin_act()
        )

        self.block3      = torch.nn.Sequential(
            torch.nn.Linear(n_z,n_z),
            torch.nn.Dropout(dropout),
            lin_act()
        )

        self.block4      = torch.nn.Sequential(
            torch.nn.Linear(n_z,n_z),
            torch.nn.Dropout(dropout),
            lin_act()
        )

        self.block5      = torch.nn.Sequential(
            torch.nn.Linear(n_z,n_z),
            torch.nn.Dropout(dropout),
            lin_act()
        )

        self.block6      = torch.nn.Sequential(
            torch.nn.Linear(n_z,n_z),
            torch.nn.Dropout(dropout),
            lin_act()
        )

        self.block7      = torch.nn.Sequential(
            torch.nn.Linear(n_z,n_z),
            torch.nn.Dropout(dropout),
            lin_act()
        )

        self.block8      = torch.nn.Sequential(
            torch.nn.Linear(n_z,n_z),
            torch.nn.Dropout(dropout),
            lin_act()
        )

        self.block9      = torch.nn.Sequential(
            torch.nn.Linear(n_z,n_z),
            torch.nn.Dropout(dropout),
            lin_act()
        )

        self.block10     = torch.nn.Sequential(
            torch.nn.Linear(n_z,n_out),
            torch.nn.Tanh()
        )


        self.expander = self.handler.constructor.clone().to(self.device)
        self.to(self.device)


    
    def forward(self,x:torch.Tensor)->torch.Tensor:
        x       = self.block1(x) + x 
        x       = self.block2(x) + x 
        x       = self.block3(x) + x 
        x       = self.block4(x) + x 
        x       = self.block5(x) + x 
        x       = self.block6(x) + x 
        x       = self.block7(x) + x 
        x       = self.block8(x) + x 
        x       = self.block9(x) + x 
        x       = self.block10(x)
        x       = x.T
       
        x       = torch.cat([self.expander @ x]).T
        return x



class ConvTransposeResBlock(torch.nn.Module):

    def __init__(self,n_ch:int,kernel_size:int,stride:int,padding:int=0,bias:bool=False,act_fn=torch.nn.LeakyReLU):
        
        super(ConvTransposeResBlock,self).__init__()

        self.block      = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(n_ch,n_ch,kernel_size=kernel_size,stride=stride,padding=padding,bias=bias),
            torch.nn.BatchNorm1d(n_ch),
            act_fn()
        )

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        return self.block(x)



class ConvResBlock(torch.nn.Module):

    def __init__(self,n_ch:int,kernel_size:int,stride:int,padding:int=0,bias:bool=False,act_fn=torch.nn.LeakyReLU):
        
        super(ConvResBlock,self).__init__()

        self.block      = torch.nn.Sequential(
            torch.nn.Conv1d(n_ch,n_ch,kernel_size=kernel_size,stride=stride,padding=padding,bias=False),
            torch.nn.BatchNorm1d(n_ch),
            act_fn(),

            torch.nn.Conv1d(n_ch,n_ch,kernel_size=kernel_size,stride=stride,padding=padding,bias=False),
            torch.nn.BatchNorm1d(n_ch),
            act_fn()
        )


    def forward(self,x:torch.Tensor) -> torch.Tensor:
        return self.block(x)

    def apply(self,fn):
        self.block.apply(fn)



class WaveGen(SoundModel):

    def __init__(self,n_z:int,sample_rate:int,upsample_rate:int,act_fn=torch.nn.LeakyReLU,mode='conv'):

        super(WaveGen,self).__init__(mode=mode,n_z=n_z)

        n_ch                    = 512
        biased                  = True

        ckernel_size            = 3

        self.input_block        = torch.nn.Sequential(

            torch.nn.ConvTranspose1d(n_z,n_ch*4,256,4,0,bias=biased),
            act_fn(),
            ConvResBlock(n_ch=n_ch*4,kernel_size=ckernel_size,stride=1,padding=(ckernel_size)//2,bias=biased,act_fn=act_fn),
            

            torch.nn.ConvTranspose1d(n_ch*4,n_ch*2,16,2,7,bias=biased),
            act_fn(),
            ConvResBlock(n_ch=n_ch*2,kernel_size=ckernel_size,stride=1,padding=(ckernel_size)//2,bias=biased,act_fn=act_fn),


            torch.nn.ConvTranspose1d(n_ch*2,n_ch,16,2,7,bias=biased),
            act_fn(),
            ConvResBlock(n_ch=n_ch,kernel_size=ckernel_size,stride=1,padding=(ckernel_size)//2,bias=biased,act_fn=act_fn),


            torch.nn.ConvTranspose1d(n_ch,n_ch//2,16,2,7,bias=biased),
            act_fn(),
            ConvResBlock(n_ch=n_ch//2,kernel_size=ckernel_size,stride=1,padding=(ckernel_size)//2,bias=biased,act_fn=act_fn),


            torch.nn.ConvTranspose1d(n_ch//2,n_ch//4,16,2,7,bias=biased),
            act_fn(),
            ConvResBlock(n_ch=n_ch//4,kernel_size=ckernel_size,stride=1,padding=(ckernel_size)//2,bias=biased,act_fn=act_fn),
        )

        ckernel_size            = 33
        self.output_blocks      = torch.nn.Sequential(
            torch.nn.Conv1d(n_ch//4,1,kernel_size=ckernel_size,stride=1,padding=ckernel_size//2,bias=True),
            torch.nn.Tanh(),


        )
        self.upsampler          = torchaudio.transforms.Resample(sample_rate,upsample_rate)
        self.to(self.device)


    def forward(self,x:torch.Tensor) -> torch.Tensor:
        x       = self.input_block(x)
        x       = self.output_blocks(x)
        x       = self.upsampler(x)
        return x



class WaveGen2(SoundModel):

    def __init__(self,n_z:int,sample_rate:int,upsample_rate:int,act_fn=torch.nn.LeakyReLU,mode='conv'):

        super(WaveGen2,self).__init__(mode=mode,n_z=n_z)

        n_ch                    = 4
        biased                  = True

        ckernel_size            = 5

        self.input_block        = torch.nn.Sequential(
            torch.nn.Linear(8,256),
            torch.nn.Dropout(p=.5),
            act_fn(),
            torch.nn.Linear(256,512),
            torch.nn.Dropout(p=.5),
            act_fn(),
            torch.nn.Linear(512,1024),
            torch.nn.Dropout(p=.5),
            act_fn(),
     

            torch.nn.ConvTranspose1d(n_z,n_z,2,2,0,bias=biased),
            act_fn(),
            ConvResBlock(n_ch=n_z,kernel_size=ckernel_size,stride=1,padding=(ckernel_size)//2,bias=biased,act_fn=act_fn),
            ConvResBlock(n_ch=n_z,kernel_size=ckernel_size,stride=1,padding=(ckernel_size)//2,bias=biased,act_fn=act_fn),


            torch.nn.ConvTranspose1d(n_z,n_z//2,2,2,0,bias=biased),
            act_fn(),
            ConvResBlock(n_ch=n_z//2,kernel_size=ckernel_size,stride=1,padding=(ckernel_size)//2,bias=biased,act_fn=act_fn),
            ConvResBlock(n_ch=n_z//2,kernel_size=ckernel_size,stride=1,padding=(ckernel_size)//2,bias=biased,act_fn=act_fn),


            torch.nn.ConvTranspose1d(n_z//2,n_z//4,2,2,0,bias=biased),
            act_fn(),
            ConvResBlock(n_ch=n_z//4,kernel_size=ckernel_size,stride=1,padding=(ckernel_size)//2,bias=biased,act_fn=act_fn),
            ConvResBlock(n_ch=n_z//4,kernel_size=ckernel_size,stride=1,padding=(ckernel_size)//2,bias=biased,act_fn=act_fn),


            torch.nn.ConvTranspose1d(n_z//4,n_z//8,2,2,0,bias=biased),
            act_fn(),
            ConvResBlock(n_ch=n_z//8,kernel_size=ckernel_size,stride=1,padding=(ckernel_size)//2,bias=biased,act_fn=act_fn),
            ConvResBlock(n_ch=n_z//8,kernel_size=ckernel_size,stride=1,padding=(ckernel_size)//2,bias=biased,act_fn=act_fn),
        )

        ckernel_size            = 65
        self.output_blocks      = torch.nn.Sequential(
            torch.nn.Conv1d(n_z//8,1,kernel_size=ckernel_size,stride=1,padding=ckernel_size//2,bias=True),
            #torch.nn.Flatten(start_dim=1),
            torch.nn.Tanh(),


        )
        self.to(self.device)


    def forward(self,x:torch.Tensor) -> torch.Tensor:
        x       = self.input_block(x)
        x       = self.output_blocks(x)
        return x

    def generate_rand(self,bs:int):
        return torch.randn(size=(bs,self.n_z,8),device=self.device,dtype=torch.float)



class GenAdapter(torch.nn.Module):

    def __init__(self,in_ch,out_ch,sf,kernel_size,act_fn):
        pass
        self.up_layers      =   torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=sf),
            torch.nn.Conv1d(in_ch,out_ch,kernel_size,1,kernel_size//2),
            act_fn()
        )

        self.interpreter    = torch.nn.Sequential(
            torch.nn.Conv1d(out_ch,1,5,1,2),
            torch.nn.Tanh()
        ) 
    
    def forward(self,x:torch.Tensor)->torch.Tensor:
        return self.up_layers(x)

    def forward_final(self,x:torch.Tensor)->torch.Tensor:
        return self.interpreter(self.up_layers(x))
    


class Discriminator(SoundModel):


    def __init__(self,act_fn=torch.nn.GELU,dropout=.1,n_kernel=8,k1=127,k2=9,k3=7):
        
        super(Discriminator, self).__init__(n_z=None)

        act_kwargs                      = {}
        bias                            = True
        n_kernels                       = n_kernel

        self.conv_layers                = torch.nn.Sequential(
            torch.nn.Conv1d(1,n_kernels,k1,4,41//2,bias=bias),                              # /8
            #torch.nn.BatchNorm1d(n_kernels),
            act_fn(**act_kwargs),

            torch.nn.Conv1d(n_kernels,n_kernels*2,k2,2,9//2,bias=bias),                     # /32
            #torch.nn.BatchNorm1d(n_kernels*2),
            act_fn(**act_kwargs),

            torch.nn.Conv1d(n_kernels*2,n_kernels*2,k2,2,9//2,bias=bias),                     # /32
            #torch.nn.BatchNorm1d(n_kernels*2),
            act_fn(**act_kwargs),

            torch.nn.Conv1d(n_kernels*2,n_kernels*4,k3,4,9//2,bias=bias),                      # /128
            #torch.nn.BatchNorm1d(n_kernels*4),
            act_fn(**act_kwargs),

            torch.nn.Conv1d(n_kernels*4,n_kernels*8,k3,2,7//2,bias=bias),                      # /256
            #torch.nn.BatchNorm1d(n_kernels*8),
            act_fn(**act_kwargs),

            torch.nn.Conv1d(n_kernels*8,n_kernels*16,k2-4,2,7//2,bias=bias),                      # /512
            #torch.nn.BatchNorm1d(n_kernels*16),
            act_fn(**act_kwargs),

            torch.nn.Conv1d(n_kernels*16,n_kernels*32,k2-4,2,5//2,bias=bias),                      # /1024
            #torch.nn.BatchNorm1d(n_kernels*32),
            act_fn(**act_kwargs),

            torch.nn.Conv1d(n_kernels*32,n_kernels*32,k2-4,2,5//2,bias=bias),                      # /2048
            #torch.nn.BatchNorm1d(n_kernels*32),
            act_fn(**act_kwargs),
            
        )

        self.lin_layers                 = torch.nn.Sequential(
            torch.nn.Flatten(1),

            torch.nn.Linear(n_kernels*(256+128),512),
            torch.nn.Dropout(dropout),

            # torch.nn.Linear(512,512),
            # torch.nn.Dropout(dropout),

            # torch.nn.Linear(512,128),
            # torch.nn.Dropout(dropout),

            torch.nn.Linear(512,1),
            torch.nn.Sigmoid()
        )
              


        self.to(self.device)


    def forward(self, x:torch.Tensor)->torch.Tensor:
        print(f"x shape in D is {x.shape}")
        if not x.shape[1] == 1:
            x   = x.unsqueeze(1) 
        x       = self.conv_layers(x)
        x       = self.lin_layers(x)

        return x 



class WAVDiscriminator(SoundModel):


    def __init__(self,act_fn=torch.nn.LeakyReLU,n_kernel=8,out_size=16384):
        
        super(WAVDiscriminator, self).__init__(n_z=None)

        act_kwargs                      = {"negative_slope":.2}
        bias                            = True
        n_kernels                       = n_kernel
        self.key                        = str(random.randint(2**6,2**16))

        self.conv_layers                = torch.nn.Sequential(
            torch.nn.Conv1d(1,n_kernels,65,8,65//2,bias=bias),                           # /16
            #torch.nn.BatchNorm1d(n_kernels),
            act_fn(**act_kwargs),

            torch.nn.Conv1d(n_kernels,n_kernels*2,33,4,33//2,bias=bias),                    # /64
            #torch.nn.BatchNorm1d(n_kernels*2),
            act_fn(**act_kwargs),

            torch.nn.Conv1d(n_kernels*2,n_kernels*4,33,4,33//2,bias=bias),                  # /256
            #torch.nn.BatchNorm1d(n_kernels*2),
            act_fn(**act_kwargs),

            torch.nn.Conv1d(n_kernels*4,n_kernels*8,17,4,17//2,bias=bias),                  # /1024
            #torch.nn.BatchNorm1d(n_kernels*4),
            act_fn(**act_kwargs),

            # torch.nn.Conv1d(n_kernels*8,n_kernels*8,17,4,17//2,bias=bias),                  # /1024
            # #torch.nn.BatchNorm1d(n_kernels*4),
            # act_fn(**act_kwargs),


            torch.nn.Conv1d(n_kernels*8,n_kernels*16,9,2,9//2,bias=bias),                  # /2048
            #torch.nn.BatchNorm1d(n_kernels*8),
            act_fn(**act_kwargs),

            torch.nn.Conv1d(n_kernels*16,n_kernels*32,5,2,5//2,bias=bias),                 # /4096
            #torch.nn.BatchNorm1d(n_kernels*16),
            act_fn(**act_kwargs),

            torch.nn.Conv1d(n_kernels*32,n_kernels*64,3,2,3//2,bias=bias),                 # /8192
            #torch.nn.BatchNorm1d(n_kernels*16),
            act_fn(**act_kwargs),



            torch.nn.Conv1d(n_kernels*64,1,4,1,0,bias=bias),

            # torch.nn.Conv1d(n_kernels*32,n_kernels*64,65,4,65//2,bias=bias),                # /16384
            # #torch.nn.BatchNorm1d(n_kernels*32),
            # act_fn(**act_kwargs),

            # torch.nn.Conv1d(n_kernels*64,n_kernels*64,65,4,65//2,bias=bias),                # /65536
            # #torch.nn.BatchNorm1d(n_kernels*32),
            # act_fn(**act_kwargs),

            # torch.nn.Conv1d(n_kernels*64,n_kernels*64,65,4,65//2,bias=bias),                # /262144
            # #torch.nn.BatchNorm1d(n_kernels*32),
            # act_fn(**act_kwargs),
            
        )

        self.lin_layers                 = torch.nn.Sequential(
            torch.nn.Flatten(1),
            # torch.nn.Linear(n_kernels*64*out_size//2048,1024),
            # torch.nn.Dropout(p=.5),
            # act_fn(**act_kwargs),

            # torch.nn.Linear(1024,1)
        )
              

        self.to(self.device)


    def forward(self, x:torch.Tensor)->torch.Tensor:
        x       = self.conv_layers(x)
        x       = self.lin_layers(x)

        return x 



class GPTDiscriminator(SoundModel):
    def __init__(self,waveform_len,act_fn=torch.nn.GELU,dropout=.1,n_kernel=8,k1=127,k2=9,k3=5):
        
        super(GPTDiscriminator, self).__init__(n_z=None)

        act_kwargs                      = {}
        bias                            = False
        n_kernels                       = n_kernel
        self.key                        = str(random.randint(2**6,2**16))


        self.conv_layers                = torch.nn.Sequential(
            torch.nn.Conv1d(1,n_kernels,k1,2,k1//2,bias=bias),                      # /2
            torch.nn.BatchNorm1d(n_kernels),
            act_fn(**act_kwargs),

            torch.nn.Conv1d(n_kernels,n_kernels*2,(k2+2),2,(k2+2)//2,bias=bias),            # /4
            torch.nn.BatchNorm1d(n_kernels*2),
            act_fn(**act_kwargs),

            torch.nn.Conv1d(n_kernels*2,n_kernels*2,k2,2,k2//2,bias=bias),          # /8
            torch.nn.BatchNorm1d(n_kernels*2),
            act_fn(**act_kwargs),

            torch.nn.Conv1d(n_kernels*2,n_kernels*2,(k2-2),2,(k2-2)//2,bias=bias),          # /8
            torch.nn.BatchNorm1d(n_kernels*2),
            act_fn(**act_kwargs),

            torch.nn.Conv1d(n_kernels*2,n_kernels*2,(k3+4),2,(k3+4)//2,bias=bias),          # /16
            torch.nn.BatchNorm1d(n_kernels*2),
            act_fn(**act_kwargs),

            torch.nn.Conv1d(n_kernels*2,n_kernels*4,(k3+2),2,(k3+2)//2,bias=bias),          # /32
            torch.nn.BatchNorm1d(n_kernels*4),
            act_fn(**act_kwargs),

            torch.nn.Conv1d(n_kernels*4,n_kernels*4,k3,2,k3//2,bias=bias),          # /64
            torch.nn.BatchNorm1d(n_kernels*4),
            act_fn(**act_kwargs),

            torch.nn.Conv1d(n_kernels*4,n_kernels*4,k3,2,k3//2,bias=bias),          # /64
            torch.nn.BatchNorm1d(n_kernels*4),
            act_fn(**act_kwargs),

            torch.nn.Conv1d(n_kernels*4,n_kernels*4,(k3-2),2,(k3-2)//2,bias=bias),          # /64
            torch.nn.BatchNorm1d(n_kernels*4),
            act_fn(**act_kwargs),

            # torch.nn.Conv1d(n_kernels*16,n_kernels*32,5,2,5//2,bias=bias),                      # /1024
            # torch.nn.BatchNorm1d(n_kernels*32),
            # act_fn(**act_kwargs),

            # torch.nn.Conv1d(n_kernels*32,n_kernels*32,5,2,5//2,bias=bias),                      # /2048
            # torch.nn.BatchNorm1d(n_kernels*32),
            # act_fn(**act_kwargs),
            
        )

        self.lin_layers                 = torch.nn.Sequential(
            torch.nn.Flatten(1),
            torch.nn.Linear((n_kernels*4) * (waveform_len//(512)),1)
           # torch.nn.Sigmoid()
        )
              


        self.to(self.device)


    def forward(self, x:torch.Tensor)->torch.Tensor:
        x       = self.conv_layers(x)
        x       = self.lin_layers(x)
        return x 
    


class DenoiseModel(torch.nn.Module):


    def __init__(self):

        super(DenoiseModel,self).__init__()

        waveform_conv_size  = 33
        inter_conv_shape    = 5
        wave_out_size       = 513
        
        
        #Maintain shape
        self.block1         = torch.nn.Sequential(
            torch.nn.Conv1d(1,8,waveform_conv_size,1,padding=waveform_conv_size//2),
            torch.nn.BatchNorm1d(8),
            torch.nn.ReLU()
        )

        #Maintain shape
        self.block2         = torch.nn.Sequential(
            torch.nn.Conv1d(8,16,inter_conv_shape,1,padding=inter_conv_shape//2),
            torch.nn.BatchNorm1d(8),
            torch.nn.ReLU()
        )

        #Maintain shape
        self.block3         = torch.nn.Sequential(
            torch.nn.Conv1d(16,8,inter_conv_shape,1,padding=inter_conv_shape//2),
            torch.nn.BatchNorm1d(8),
            torch.nn.ReLU()
        )

        #Maintain shape
        self.block4         = torch.nn.Sequential(
            torch.nn.Conv1d(8,1,wave_out_size,1,padding=wave_out_size//2),
            torch.nn.Tanh()
        )
    

    def forward(self,x:torch.Tensor)->torch.Tensor:
        x           = self.block1(x)
        x           = self.block2(x)
        x           = self.block3(x)
        x           = self.block4(x)
        return      


class Decoder(torch.nn.Module):

    def __init__(self):

        super(Decoder,self).__init__()

        self.convLayers     = torch.nn.Sequential(
            torch.nn.Conv2d(3,16,3,1,1),
            torch.nn.LeakyReLU(.2),
            torch.nn.Conv2d(16,32,3,1,1),
            torch.nn.LeakyReLU(.2),
            torch.nn.AvgPool2d(2),


            torch.nn.Conv2d(32,64,3,1,1),
            torch.nn.LeakyReLU(.2),
            torch.nn.Conv2d(64,128,5,1,2),
            torch.nn.LeakyReLU(.2),
            torch.nn.AvgPool2d(2),


            torch.nn.Conv2d(128,256,5,1,2),
            torch.nn.LeakyReLU(.2),
            torch.nn.Conv2d(256,512,5,1,2),
            torch.nn.LeakyReLU(.2),
            torch.nn.AvgPool2d(2),


            torch.nn.Conv2d(512,512,7,1,3),
            torch.nn.LeakyReLU(.2),
            torch.nn.AvgPool2d(2),


            #Upwards
            torch.nn.Flatten(start_dim=2),
            torch.nn.Linear(1560,1024),
            torch.nn.LeakyReLU(.2),
            torch.nn.Linear(1024,1024),
            #torch.nn.Unflatten(dim=1,unflattened_size=(512,1024)),

            # # torch.nn.Conv1d(1024,1024,5,1,2),
            # # torch.nn.LeakyReLU(.2),


            # # torch.nn.Upsample(size=128),
            # # torch.nn.Conv1d(1024,512,5,1,2),
            # # torch.nn.LeakyReLU(.2),

            # torch.nn.Upsample(size=2048),
            # torch.nn.Conv1d(512,256,5,1,2),
            # torch.nn.LeakyReLU(.2),

            # torch.nn.Upsample(size=1024),
            # torch.nn.Conv1d(256,128,5,1,2),
            # torch.nn.LeakyReLU(.2),

            torch.nn.Upsample(size=2048),
            torch.nn.Conv1d(512,256,5,1,2),
            torch.nn.LeakyReLU(.2),

            torch.nn.Upsample(size=4096),
            torch.nn.Conv1d(256,64,5,1,2),
            torch.nn.LeakyReLU(.2),

            torch.nn.Upsample(size=8192),
            torch.nn.Conv1d(64,16,5,1,2),
            torch.nn.LeakyReLU(.2),

            torch.nn.Upsample(size=16384),
            torch.nn.Conv1d(16,1,5,1,2),
            torch.nn.Tanh(),
            torch.nn.Flatten(start_dim=1,end_dim=-1)

        )

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        x       = self.convLayers(x)
        return x


if __name__ == "__main__":

    #m = WAVDiscriminator(n_kernel=32)
    #print(f"out shape is {m.forward(torch.randn(size=(8,1,8192),device=torch.device('cuda'))).shape}")
    
    m   = WaveGen2(128,1024,1024)
    print(f"out shape is {m.forward(torch.randn(size=(8,128,4),device=torch.device('cuda'))).shape}")

