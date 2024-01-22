import torch
import random
import math
import numpy



def weights_initG(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.kaiming_uniform_(m.weight.data,.2)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.kaiming_uniform_(m.weight.data,.3)
        torch.nn.init.constant_(m.bias.data, 0)


def weights_initD(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data,0,.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data,1,.02)
        torch.nn.init.constant_(m.bias.data, 0)


def print_epoch_header(epoch_num,epoch_tot,header_width=100):
    print("="*header_width)
    epoch_seg   = f'=   EPOCH{f"{epoch_num+1}/{epoch_tot}".rjust(8)}'
    print(epoch_seg,end="")
    print(" "*(header_width-(len(epoch_seg)+1)),end="=\n")
    print("="*header_width)


def model_size(model:torch.nn.Module):
    return f"{ (sum([p.numel()*p.element_size() for p in model.parameters()])/(1024*1024)):.2f}MB"
