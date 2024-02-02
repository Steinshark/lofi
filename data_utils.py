import torch
from torch.utils.data import Dataset
import math
import random
import numpy
import time
import torchaudio
import os
from matplotlib import pyplot as plt
import json

def save_waveform_wav(x:torch.Tensor,filename:str,sample_rate:int):
    waveform        = torchaudio.transforms.Resample(sample_rate,44100)(x)
    torchaudio.save(filename,torch.stack([waveform,waveform]),44100)

def mp3_to_wav(path:str="C:/data/music/mp3",save_path:str="C:/data/music/wav"):

    for fname in [os.path.join(path,f) for f in os.listdir(path)]:
        waveform    = torchaudio.load(fname)[0] 
        torchaudio.save(fname.replace("mp3","wav"),waveform,44100)

def reduce_arr(arr,newlen):

    #Find GCF of len(arr) and len(newlen)
    gcf         = math.gcd(len(arr),newlen)
    mult_fact   = int(newlen / gcf) 
    div_fact    = int(len(arr) / gcf) 

    new_arr     = numpy.repeat(arr,mult_fact)


    return [sum(list(new_arr[n*div_fact:(n+1)*div_fact]))/div_fact for n in range(newlen)]

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.025)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)


class PCA_Handler:


    def __init__(self,from_wav_folder:str="",from_vectors:str='',non_pca:bool=False,sample_rate=4096):
        random.seed(512)
        torch.manual_seed(512)
        self.device     = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.non_pca    = non_pca
        self.sample_rate        = sample_rate
        
        if non_pca:
            pass

        #Create from references to U,S,V 
        elif not from_vectors == '':
            self.U      = torch.load(from_vectors+"U.pt")
            self.S      = torch.load(from_vectors+"S.pt")
            self.V_t    = torch.load(from_vectors+"V_t.pt")
            #Get constructor vector
            self.constructor= self.U @ self.S 
            self.cuda_constructor   = self.constructor.clone().to(self.device)
            self.U_cuda             = self.U.clone().to(self.device)
            self.pca_rank           = self.V_t.shape[0]
            print(f"pca rank is {self.pca_rank}")
        elif not from_wav_folder == '':
            pass


    def construct_pca_from_wavs(self,load_path:str="C:/data/music/wav/",sample_rate:int=4096,length_s:int=16,pca_rank:int=64,n_samples:int=512):


        #Create a resample object to convert from 44100 to sample_rate
        downsampler             = torchaudio.transforms.Resample(44100,sample_rate)

        #Generate save path 
        save_path               = f"C:/data/music/{sample_rate}_{pca_rank}_{length_s}/"
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
            split_probs         = [1 for _ in range(waveform_len-window)]
            for i in range(int(.2*len(split_probs))):
                split_probs[i]  = .1 
                split_probs[-i] = .1

            s                   = sum(split_probs)
            split_probs         = [i/s for i in split_probs]
            sample_splits       = random.choices(list(range(waveform_len-window)),split_probs,k=n_samples)

            #Generate data matrix
            for i in sample_splits:

                #Sample vector
                data_vector         = first_channel_audio[i:i+window].clone()

                assert data_vector.shape[-1] == window
                data_vectors.append(data_vector)


        #Compute PCA from vectors 
        random.shuffle(data_vectors)
        data_matrix             = torch.stack(data_vectors).T#.float().T
        data_matrix             = data_matrix - data_matrix.mean()
        U,S,V                   = torch.pca_lowrank(data_matrix,q=pca_rank,niter=3)

        self.U                  = U
        self.S                  = S.diag()
        self.V_t                = V.T

        self.pca_rank           = pca_rank
        self.sample_rate        = sample_rate
        self.constructor        = self.U @ self.S 
        self.constructor_cuda   = self.constructor.clone().to(self.device)
        self.U_cuda             = self.U.clone().to(self.device)

        #Save 
        torch.save(self.U,save_path+"U.pt")
        torch.save(self.S,save_path+"S.pt")
        torch.save(self.V_t,save_path+"V_t.pt")

        #Get a sample
        save_waveform_wav(self.expand(self.V_t[:,56]),f"C:/gitrepos/lofi/wavdump/test+rank{pca_rank}_sr{sample_rate}.wav",sample_rate=sample_rate)
        return


    def ds_no_pca(self,load_path:str="C:/data/music/wav/",length_s:int=16,n_samples:int=512):
        
        #Create a resample object to convert from 44100 to sample_rate
        downsampler             = torchaudio.transforms.Resample(44100,self.sample_rate)

        #Define sample data vars 
        window                  = length_s * self.sample_rate

        #Generate data matrix from wav
        self.data_vectors            = []
        for fname in os.listdir(load_path):

            #Load waveform tensor
            filename            = load_path + fname 
            first_channel_audio = downsampler(torchaudio.load(filename)[0])[0]


            #PROCESS

            #remove first 20 and last 10 
            first_channel_audio = first_channel_audio[self.sample_rate*20:-self.sample_rate*10]
            waveform_len        = first_channel_audio.shape[0]

            #Indices
            actual_samples      = min(n_samples,int(n_samples * (waveform_len/(self.sample_rate*3*60))))
            probs               = numpy.asarray(list(range(waveform_len-window)))
            probs               = probs * 2 * numpy.pi / (waveform_len-window)
            probs               = [abs(numpy.sin(i))**.2 for i in probs]


            sample_splits       = random.choices(list(range(waveform_len-window)),probs,k=actual_samples)

            #Generate data matrix
            for i in sample_splits:

                #Sample vector
                data_vector         = first_channel_audio[i:i+window].clone()
                # data_vector         = data_vector - data_vector.mean()
                data_vector         = data_vector.clip(min=-.8,max=.8) / .8

                assert data_vector.shape[-1] == window
                self.data_vectors.append(data_vector)
        
        #SHUFFLE 
        #random.shuffle(self.data_vectors)
        save_waveform_wav(random.choice(self.data_vectors),f"C:/gitrepos/lofi/wavdump/test_rank_NA_sr{self.sample_rate}.wav",sample_rate=self.sample_rate)
        self.pca_rank       = 'N/A'
        self.max_load       = len(self.data_vectors)


    def yield_data_from_matrix(self,i:int):
        if self.non_pca:
            return self.data_vectors[i]
        with torch.no_grad():
            return self.constructor @ self.V_t[:,i]


    def expand(self,x:torch.Tensor,mode='cpu')->torch.Tensor:
        #print(f"mult is {self.constructor.shape} @ {x.shape}")
        if mode == 'cuda':
            return self.constructor_cuda @ x 
        
        return self.constructor @ x 


    def compress(self,x:torch.Tensor,mode='cpu')->torch.Tensor:
        #print(f"mult is {x.shape} @ {self.U.shape}")
        if mode == 'cuda':
            return x @ self.U_cuda
        return x @ self.U

    def __len__(self):
        if self.non_pca:
            return self.max_load
        else:
            return self.V_t.shape


class Stats_Handler:

    def __init__(self,models:list):

        self.models             = models 

        self.errorsD            = {model.key: {   "real":{'batch':[],'all':[]},
                                                    "fake":{'batch':[],'all':[]}}
                                    for model in models}
        self.errorsG            = {model.key: {   "fake":{'batch':[],'all':[]}}
                                    for model in models}
        
        self.classifications    = {model.key: {   "real":{'batch':[],'all':[]},
                                                    "fake":{'batch':[],'all':[]},
                                                    "rand":{'batch':[],'all':[]}}
                                    for model in models
                                    }
    
        self.epoch_num          = 0
    
    def add_real(self,key,error=None,classification=None):
        if not error is None:
            self.errorsD[key]['real']['batch'].append(error)
            self.errorsD[key]['real']['all'].append(error)
        elif not classification is None:
            self.classifications[key]['real']['batch'].append(classification)
            self.classifications[key]['real']['all'].append(classification)

    
    def add_fakeD(self,key,error=None,classification=None):
        if not error is None:
            self.errorsD[key]['fake']['batch'].append(error)
            self.errorsD[key]['fake']['all'].append(error)
        elif not classification is None:
            self.classifications[key]['fake']['batch'].append(classification)
            self.classifications[key]['fake']['all'].append(classification)

    def add_fakeG(self,key,error):
        self.errorsG[key]['fake']['batch'].append(error)
        self.errorsG[key]['fake']['all'].append(error)
    
    def step_ep(self):

        for key in [m.key for m in self.models]:
            self.errorsD[key]['real']['batch']   = [] 
            self.errorsD[key]['fake']['batch']   = [] 
            
            self.errorsG[key]['fake']['batch']   = [] 

            self.classifications[key]['real']['batch']   = [] 
            self.classifications[key]['fake']['batch']   = [] 
            self.classifications[key]['rand']['batch']   = [] 

    def add_rand(self,key,classifications):
        self.classifications[key]['rand']['batch'].append(classifications)
        self.classifications[key]['rand']['all'].append(classifications)

    def get_d_real(self,key,mode='all',returntype='mean'):
        if returntype  == 'mean':
            return sum(self.classifications[key]['real'][mode]) / len(self.classifications[key]['real'][mode])
        else:
            return self.classifications[key]['real'][mode]
    
    def get_d_fake(self,key,mode='all',returntype='mean'):
        if returntype  == 'mean':
            return sum(self.classifications[key]['fake'][mode]) / len(self.classifications[key]['fake'][mode])
        else:
            return self.classifications[key]['fake'][mode]
    
    def get_g_err(self,key,mode='all',returntype='mean'):
        if returntype  == 'mean':
            return sum(self.errorsG[key]['fake'][mode]) / len(self.errorsG[key]['fake'][mode])
        else:
            return self.errorsG[key]['fake'][mode]
    
    def get_d_real_err(self,key,mode='all',returntype='mean'):
        if returntype  == 'mean':
            return sum(self.errorsD[key]['real'][mode]) / len(self.errorsD[key]['real'][mode])
        else:
            return self.errorsD[key]['real'][mode]
    
    def get_d_fake_err(self,key,mode='all',returntype='mean'):
        if returntype  == 'mean':
            return sum(self.errorsD[key]['fake'][mode]) / len(self.errorsD[key]['fake'][mode])
        else:
            return self.errorsD[key]['fake'][mode]
    
    def get_rand(self,key,mode='all',returntype='mean'):
        if returntype  == 'mean':
            return sum(self.classifications[key]['rand'][mode]) / len(self.classifications[key]['rand'][mode])
        else:
            return self.classifications[key]['rand'][mode]


    def save_state(self,savepath:str):

        with open(os.path.join(savepath,"data",'handler.stat'),'w') as file:
            savedict            = {'errorsD':self.errorsD,'errorsG':self.errorsG,'classifications':self.classifications,'epoch_num':self.epoch_num}

            file.write(json.dumps(savedict))


    def load_state(self,savepath:str):
        with open(os.path.join(savepath,"data",'handler.stat'),'r') as file:
            savedict            = json.loads(file.read())

            self.errorsD            = savedict['errorsD']
            self.errorsG            = savedict['errorsG']
            self.classifications    = savedict['classifications']
            self.epoch_num          = savedict['epoch_num']


#AudioDataSet takes a PCA_handler and will generate data in the __getitem__ method via these 3 
class AudioDataSet(Dataset):


    def __init__(self,pca_handler:PCA_Handler):
        self.handler        = pca_handler
        #self.data           = []
        if self.handler.non_pca:
            pass 
        else:
            self.matrix         = pca_handler.U @ pca_handler.S @ pca_handler.V_t
            matrix_n            = self.matrix.T.numpy()

        # for i in range(len(matrix_n)):
        #     tensor      = torch.from_numpy(matrix_n[i])
        #     self.data.append(tensor)


    def __len__(self):
        return (len(self.handler))


    def __getitem__(self,i):
        #returner    = self.data[i]
        #print(f"returning shape {returner.shape}")
        #return self.data[i]
        return self.handler.yield_data_from_matrix(i)

    
    def __repr__(self):
        return f"<ADS len{self.__len__()}>"



if __name__ == "__main__":
    #mp3_to_wav()
    # torch.manual_seed(512)
    # random.seed(512)
    # p   = PCA_Handler(from_wav_folder="C:/data/music/wavs/")
    # sr  = 2048
    # for pca_rank in [4096]:
    #     p.construct_pca_from_wavs(pca_rank=pca_rank,n_samples=256,sample_rate=sr)
    #     data    = p.yield_data_from_matrix(712)
    #     print(f"max is {torch.max(data)},mion is {torch.min(data)},size is {data.shape}, t={int(data.shape[-1]/sr)}")

    a = PCA_Handler(non_pca=True,sample_rate=512)
    a.ds_no_pca(sample_rate=512,n_samples=2048)
