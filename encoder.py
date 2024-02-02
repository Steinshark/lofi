import torch 
import torchaudio
import data_utils 
from matplotlib import pyplot as plt
import PIL
from PIL import Image
import io
from networks import Decoder
import torchvision 
from torch.utils.data import Dataset,DataLoader 
import os 

spectrogrammer  = torchaudio.transforms.Spectrogram(n_fft=128)
imager          = torchvision.transforms.ToTensor()

class AudioDS(Dataset):

    def __init__(self,root):
        self.data   = []

        for f in set([fname.replace(".pt",'').replace(".jpg",'') for fname in os.listdir(root)]):

            f   = os.path.join(root,f)

            x   = imager(Image.open(os.path.join(root,f"{f}.jpg")))
            y   = torch.load(os.path.join(root,f"{f}.pt"))

            self.data.append((x,y))
    
    def __getitem__(self,i:int):
        return self.data[i]
    
    def __len__(self):
        return len(self.data)


def generate_spectrogram(waveform:torch.Tensor,i:int):
    spec        = spectrogrammer(waveform)
    
    #save
    # print(f"dims are {spec.shape}")
    plt.subplots(figsize=(16, 8))
    plt.axis('off')
    plt.imshow(spec)
    plt.savefig(f"C:/data/music/specs/{i}.jpg",bbox_inches='tight')
    plt.close()

    return spec

def spec_to_img(spectrogram:torch.Tensor):
    return 0


def train(model:torch.nn.Module):

    dataset     = AudioDS("C:/data/music/specs/")
    dataloader  = DataLoader(dataset,8,True)


    err         = torch.nn.MSELoss()
    optim       = torch.optim.Adam(model.parameters(),lr=.001)

    for _ in range(10):

        for i,batch in enumerate(dataloader):
            
            for param in model.parameters():
                param.grad = None 

            x       = batch[0].cuda()
            y       = batch[1].cuda()

            y_pred  = model.forward(x)

            loss    = err(y,y_pred)
            loss.backward()


            optim.step()

        print(f"loss={loss.mean():.4f}")
        data_utils.save_waveform_wav(y_pred[0].detach().cpu(),f"C:/data/music/runs/run{_}.wav",sample_rate=1024)



if __name__ == "__main__":
    d           = Decoder().cuda()
    handler     = data_utils.PCA_Handler(non_pca=True,sample_rate=1024)
    handler.ds_no_pca(n_samples=4)

    train(d)

    # for i in range(len(handler.data_vectors)):
    #     s           = generate_spectrogram(handler.data_vectors[i],i)
    #     torch.save(handler.data_vectors[i],f"C:/data/music/specs/{i}.pt")
    #     # img         = imager(Image.open(f"C:/data/music/specs/{i}.jpg"))
    #     # print(f"img is {img.shape}")
    #     # g           = d.forward(img)
    #     # print(f"out is {g.shape}")
