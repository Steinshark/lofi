import torch
from torch.utils.data import Dataset
import math
import random
import numpy
import time
import torchaudio
import os
from matplotlib import pyplot as plt



def save_waveform_wav(x:torch.Tensor,filename:str,sample_rate:int):
    waveform        = torchaudio.transforms.Resample(sample_rate,44100)(x)
    torchaudio.save(filename,torch.stack([waveform,waveform]),44100)

def mp3_to_wav(path:str="C:/data/music/mp3",save_path:str="C:/data/music/wav"):

    for fname in [os.path.join(path,f) for f in os.listdir(path)]:
        waveform    = torchaudio.load(fname)[0] 
        torchaudio.save(fname.replace("mp3","wav"),waveform,44100)

class PCA_Handler:


    def __init__(self,from_wav_folder:str="",from_vectors:list[torch.Tensor]=None):

        #Create from references to U,S,V 
        if not from_vectors is None:
            self.U      = from_vectors[0]
            self.S      = from_vectors[1]
            self.V      = from_vectors[2]
            #Get constructor vector
            self.constructor= self.U @ self.S 

        elif not from_wav_folder == '':
            pass


    def construct_pca_from_wavs(self,load_path:str="C:/data/music/wavs/",sample_rate:int=4096,length_s:int=16,pca_rank:int=64,n_samples:int=512):


        #Create a resample object to convert from 44100 to sample_rate
        downsampler             = torchaudio.transforms.Resample(44100,sample_rate)

        #Generate save path 
        save_path               = f"C:/data/music/{sample_rate}_{pca_rank}_{length_s}"
        if not os.path.isdir(save_path):
                os.mkdir(save_path)

        #Define sample data vars 
        window                  = length_s * sample_rate

        #Generate data matrix from wav
        data_vectors            = []
        for fname in os.listdir(load_path):

            #Load waveform tensor
            filename            = load_path + fname 
            first_channel_audio = downsampler(torchaudio.load(filename)[0])[0] 
            first_channel_audio = first_channel_audio - first_channel_audio.mean()
            waveform_len        = first_channel_audio.shape[0]

            #Indices
            sample_splits       = random.sample(list(range(waveform_len-window)),k=n_samples)

            #Generate data matrix
            for i in sample_splits:

                #Sample vector
                data_vector         = first_channel_audio[i:i+window].clone()

                assert data_vector.shape[-1] == window
                data_vectors.append(data_vector)


        #Compute PCA from vectors 
        #random.shuffle(data_vectors)
        data_matrix             = torch.stack(data_vectors).T#.float().T
        data_matrix             = data_matrix - data_matrix.mean()
        self.U,self.S,self.V    = torch.pca_lowrank(data_matrix,q=pca_rank,niter=8)

        self.S                  = torch.diag(self.S)
        self.V_t                = self.V.T

        self.constructor= self.U @ self.S 
        return


    def yield_data_from_matrix(self,i:int):
        return self.constructor @ self.V_t[:,i]


    def expand(self,x:torch.Tensor)->torch.Tensor:
        print(f"mult:{self.constructor.shape}, {x.shape}")
        return self.constructor @ x 


    def compress(self,x:torch.Tensor)->torch.Tensor:
        print(f"mult:{x.shape}, {self.U.shape}")
        return x @ self.U



#AudioDataSet takes a PCA_handler and will generate data in the __getitem__ method via these 3 
class AudioDataSet(Dataset):

    def __init__(self,pca_handler:PCA_Handler):
        
        self.handler        = pca_handler
            
    def __len__(self):
        return self.V.shape[1]

    def __getitem__(self,i):
        return self.handler.yield_data_from_matrix(i)
    
    def __repr__(self):
        return f"<ADS len{self.__len__()}>"



if __name__ == "__main__":
    torch.manual_seed(512)
    random.seed(512)
    p   = PCA_Handler(from_wav_folder="C:/data/music/wavs/")
    sr  = 2048
    for pca_rank in [2048+1024]:
        p.construct_pca_from_wavs(pca_rank=pca_rank,n_samples=256,sample_rate=sr)
        data    = p.yield_data_from_matrix(712)
        print(f"max is {torch.max(data)},mion is {torch.min(data)},size is {data.shape}, t={int(data.shape[-1]/sr)}")
        save_waveform_wav(data,f"C:/gitrepos/lofi/wavdump/test{sr}_{pca_rank}.wav",sr)

    