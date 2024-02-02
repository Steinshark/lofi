import torch 
from collections import OrderedDict
from networks import *
from torch.utils.data import Dataset, DataLoader
import numpy 
import time 
import json
import os 
import random 
import sys 
import pprint 
from network_utils import weights_initG,weights_initD, print_epoch_header,model_size
from torch.optim import Adam 
from hashlib import md5
from matplotlib import pyplot as plt 
from torch.profiler import record_function
from scipy.stats import linregress
from matplotlib import pyplot as plt 
import networks
import data_utils
import math
from torch import backends
from torch.backends.cudnn import benchmark as cudnn_benchmark
import torchaudio
#Provides the mechanism to train the model out
class Trainer:

    #Initialize
    def __init__(self,modelG:SoundModel,modelDs:list[SoundModel],pca_handler:data_utils.PCA_Handler,resume:bool=False):

        #Prepare models
        self.modelG                         = modelG 
        self.d_models:list[torch.nn.Module] = modelDs
        
        #Prepare stats and data
        self.pca_handler                    = pca_handler
        self.stats_handler                  = data_utils.Stats_Handler(self.d_models)

        #Set vars
        self.device                         = self.modelG.device
        self.n_z                            = self.modelG.n_z
        self.epoch_num                      = 0 

        #Set tracking and config stuff
        self.plots                          = []     
        self.set_colors()



    #Get incoming data into a dataloader
    def build_dataset(self,batch_size:int,shuffle:bool,num_workers:int):

        #Save parameters 
        self.batch_size     = batch_size

        #Create sets
        self.dataset        = data_utils.AudioDataSet(self.pca_handler)
        self.dataloader     = DataLoader(self.dataset,batch_size=batch_size,shuffle=shuffle,num_workers=1,pin_memory=False)


    def save_model_states(self,series_path:str): 
        
        modelroot       = os.path.join(series_path,'models')
        #Ensure path is good 
        if not os.path.exists(modelroot):
            os.mkdir(modelroot)

        #Save model params to file 
        torch.save(     self.modelG.state_dict(),        os.path.join(modelroot,"GMODEL.pt"))

        for model in self.d_models:
            torch.save(     model.state_dict(),    os.path.join(modelroot,f"DMODEL{model.key}.pt")) 
        
        ##Save configs
        #with open(os.path.join(path,f"{G_name}_config"),"w") as config_file:
        #    config_file.write(json.dumps(self.G_config))
        #    config_file.close()

        #with open(os.path.join(path,f"{D_name}_config"),"w") as config_file:
        #    config_file.write(json.dumps(self.D_config))
        #    config_file.close()


    #Import saved models 
    def load_models(self,series_path:str):
        
        #Create models
        #self.create_models(D_config,G_config)

        #Load params
        try:
            self.modelG.load_state_dict(     torch.load(f"{os.path.join(series_path,'models')}GMODEL.pt"))

            for model in self.d_models:

                model.load_state_dict( torch.load(f"{os.path.join(series_path,'models')}DMODEL{model.key}.pt"))
            print(f"loaded for epoch {self.epoch_num}")
        except FileNotFoundError:
            return



    #Set optimizers and error function for models
    def set_learners(self,D_optims,G_optim,error_fn):
        self.d_optims   = D_optims
        self.G_optim    = G_optim
        self.error_fn   = error_fn


    #Train Vanilla
    def train(self,verbose=True):
        t_start = time.time()
        torch.backends.cudnn.benchmark = True


        #Telemetry
        if verbose:
            width       = 100
            num_equals  = 50
            indent      = 4 
            n_batches   = len(self.dataset) // self.batch_size
            printed     = 0


            print(" "*indent,end='')
            prefix = f"{int(n_batches)} batches  Progress" 
            print(f"{prefix}",end='')
            print(" "*(width-indent-len(prefix)-num_equals-2),end='')
            print("[",end='')

            

        #Run all batches
        for i, data in enumerate(self.dataloader,0):
            
             #Prep real values
            x_set                                           = data.to(self.device)
            x_len                                           = len(x_set)

            #Use label spread
            real_fill_vals                                  = [.9+random.random()*.1 for _ in range(x_len)]
            y_set                                           = torch.ones(size=(x_len,1),dtype=torch.float,device=self.device)
            for j, val in enumerate(real_fill_vals):
                y_set[j,:] = val

            #Generate samples
            random_inputs                                   = self.modelG.generate_rand(bs=x_len)

            for model_d,d_optim in zip(self.d_models,self.d_optims):

                if verbose:
                    percent = i / n_batches
                    while (printed / num_equals) < percent:
                        print("-",end='',flush=True)
                        printed+=1

                #####################################################################
                #                           TRAIN REAL                              #
                #####################################################################
                
                #Zero First
                for param in model_d.parameters():
                    param.grad  = None

                #Classify real set
                real_class                                      = model_d.forward(x_set)
                
                #Calc error
                d_error_real                                    = self.error_fn(real_class,y_set)
                self.stats_handler.add_real(model_d.key,error=d_error_real.detach().cpu().mean().item())
                self.stats_handler.add_real(model_d.key,classification=real_class.detach().cpu().mean().item())
                d_error_real.backward()
                
                #####################################################################
                #                           TRAIN FAKE                              #
                #####################################################################

                #Classify fake samples 
                fake_fill_vals                                  = [random.random()*.1 for _ in range(x_len)]
                fake_labels                                     = torch.zeros(size=(x_len,1),dtype=torch.float,device=self.device)
                #Use label spread
                for j, val in enumerate(fake_fill_vals):
                    fake_labels[j,:] = val

                generator_outputs                               = self.modelG(random_inputs)
                fake_class                                      = model_d.forward(generator_outputs.detach())

                #Calc error
                d_error_fake                                    = self.error_fn(fake_class,fake_labels)
                d_error_fake.backward()
                self.stats_handler.add_fakeD(model_d.key,error=d_error_fake.detach().cpu().mean().item())
                self.stats_handler.add_fakeD(model_d.key,classification=fake_class.detach().cpu().mean().item())

                #####################################################################
                #                            GET RAND                               #
                #####################################################################
                with torch.no_grad():
                    model_d.eval()
                    for _ in range(generator_outputs.shape[0]):
                        random_vect             = torch.randn(size=(1,generator_outputs.shape[2]),dtype=torch.float,device=self.device)
                        abs_mult                = max(abs(torch.min(random_vect)),abs(torch.max(random_vect)))
                        random_vect             = random_vect / abs_mult
                        d_random                = model_d.forward(random_vect).cpu().detach()
                        self.stats_handler.add_rand(model_d.key,d_random.item())
                    model_d.train()


                #Step
                torch.nn.utils.clip_grad_norm_(model_d.parameters(),self.clip_to_d)
                d_optim.step()      

                

            
                #####################################################################
                #                           TRAIN GENR                              #
                #####################################################################

                #Zero Grads
                for param in self.modelG.parameters():
                    param.grad  = None


                #Classify the fakes again after Discriminator got updated 
                #Use label spread
                real_fill_vals                                  = [.9+random.random()*.1 for _ in range(x_len)]
                y_set                                           = torch.ones(size=(x_len,1),dtype=torch.float,device=self.device)
                for j, val in enumerate(real_fill_vals):
                    y_set[j,:] = val
                fake_classd                                     = model_d.forward(generator_outputs.clone())
                g_error                                         = self.error_fn(fake_classd,y_set)
                self.stats_handler.add_fakeG(model_d.key,g_error.detach().cpu().mean().item())
                
                #Back Propogate
                g_error.backward()
                torch.nn.utils.clip_grad_norm_(self.modelG.parameters(),self.clip_to_g)
                self.G_optim.step()


                #####################################################################
                #                          TRAIN SIMLIAR                            #
                #####################################################################
                #Zero Grads
                for param in self.modelG.parameters():
                    param.grad  = None
                rand_for_train          = self.modelG.generate_rand(bs=2)
                generator_outputs       = self.modelG(rand_for_train)           
                sim_err                 = self.difference_err(rand_for_train[0],rand_for_train[1],generator_outputs[0],generator_outputs[1])
                sim_err.backward()
                torch.nn.utils.clip_grad_norm_(self.modelG.parameters(),self.clip_to_g)
                self.G_optim.step()           

                
                


        if verbose:
            percent = (i+1) / n_batches
            while (printed / num_equals) < percent:
                print("-",end='',flush=True)
                printed+=1
        


        #TELEMETRY

        print(f"]")
        print("\n")

        for i,key in enumerate([d.key for d in self.d_models]):
            
            out_2 = f"D(real)={self.stats_handler.get_d_real(key,mode='batch'):.4f}    D(gen)={self.stats_handler.get_d_fake(key,mode='batch'):.4f}    D(rand)={self.stats_handler.get_rand(key,mode='batch'):.4f}"
            print(" "*(width-len(out_2)),end='')
            print(f"\nD Model {i}")
            
            print(" "*(width-len(out_2)),end='')
            print(out_2)

            out_3 = f"er_real={self.stats_handler.get_d_real_err(key,mode='batch'):.4f}     er_fke={self.stats_handler.get_d_fake_err(key,mode='batch'):.4f}    g_error={self.stats_handler.get_g_err(key,mode='batch'):.3f}"
            print(" "*(width-len(out_3)),end='')
            print(out_3)
            
            out_4   = f"rand_vect min={torch.min(random_vect[0]).item():.4f}        rand_vect max={torch.max(random_vect[0]).item():.4f}"
            print(" "*(width-len(out_4)),end='')
            print(out_4)

        if (d_error_real < .0001) and ((d_error_fake) < .0001):
            return True
        
        self.stats_handler.step_ep()


    #Train Vanilla
    def train_mp(self,verbose=True):
        Dscaler          = torch.cuda.amp.GradScaler()
        Gscaler          = torch.cuda.amp.GradScaler()
        

        #Telemetry
        if verbose:
            width       = 100
            num_equals  = 50 // len(self.d_models)
            indent      = 4 
            n_batches   = len(self.dataset) // self.batch_size
            printed     = 0


            print(" "*indent,end='')
            prefix = f"{int(n_batches)} batches  Progress" 
            print(f"{prefix}",end='')
            print(" "*(width-indent-len(prefix)-(num_equals*len(self.d_models))-2),end='')
            print("[",end='')

            

        #Run all batches
        for model_d,d_optim in zip(self.d_models,self.d_optims):

            for i, data in enumerate(self.dataloader,0):

                #Prep real values
                x_set:torch.Tensor                              = data.to(self.device).unsqueeze(1) 
                x_len                                           = len(x_set)

                if verbose:
                    percent = i / n_batches
                    while (printed / num_equals) < percent:
                        print("-",end='',flush=True)
                        printed+=1


                #prep models
                self.modelG.eval()
                for param in model_d.parameters():
                    param.grad  = None



                #Autocast real classification
                with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=True):
                    #####################################################################
                    #                           TRAIN REAL                              #
                    #####################################################################

                    #Generate real, noisy labels
                    real_labels                                     = torch.randn(x_set.shape[0],dtype=torch.float,device=self.modelG.device)
                    real_labels                                     = (.05 * real_labels / real_labels.max()) + .9
                    real_labels                                     = real_labels.unsqueeze_(dim=-1)
                  
                    #Classify real set
                    real_class                                      = model_d.forward(x_set)
                    
                    #Calc error
                    d_error_real                                    = self.error_fn(real_class,real_labels)

                #Update parameters
                Dscaler.scale(d_error_real).backward()
                #Dscaler.unscale_(d_optim)
                #torch.nn.utils.clip_grad_norm_(model_d.parameters(), self.max_norm_d)
                #Dscaler.step(d_optim)
                #Dscaler.update()



                #Autocast fake classification
                with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=True):
                    #####################################################################
                    #                           TRAIN FAKE                              #
                    #####################################################################

                    #Generate fake labels
                    fake_labels                                     = torch.zeros(size=real_labels.shape,dtype=torch.float,device=self.device)

                    #Classify fake set 
                    random_inputs                                   = self.modelG.generate_rand(bs=x_len)
                    generator_outputs                               = self.modelG(random_inputs)
                    fake_class                                      = model_d.forward(generator_outputs)

                    #Calc error
                    d_error_fake                                    = self.error_fn(fake_class,fake_labels)

                #Update parameters
                Dscaler.scale(d_error_fake).backward()
                Dscaler.unscale_(d_optim)
                torch.nn.utils.clip_grad_norm_(model_d.parameters(), self.max_norm_d)
                Dscaler.step(d_optim)
                Dscaler.update()
                


                # #Autocast fake classification
                # with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=True):
                #     #####################################################################
                #     #                           TRAIN RAND                              #
                #     #####################################################################

                #     #Generate fake labels
                #     fake_labels                                     = torch.zeros(size=real_labels.shape,dtype=torch.float,device=self.device)

                #     #Classify rand set 
                #     random_outputs                                  = torch.randn(size=x_set.shape,dtype=torch.float,device=self.modelG.device)
                #     random_outputs                                  = random_outputs - random_outputs.min(-1,keepdim=True)[0]
                #     random_outputs                                  = 2*(random_outputs / random_outputs.max(-1,keepdim=True)[0]) - 1

                #     rand_class                                      = model_d.forward(random_outputs)

                #     #Calc error
                #     d_error_rand                                    = self.error_fn(rand_class,fake_labels)

                # #Update parameters
                # Dscaler.scale(d_error_rand).backward()
                # Dscaler.unscale_(d_optim)
                # torch.nn.utils.clip_grad_norm_(model_d.parameters(), self.max_norm_d)
                # Dscaler.step(d_optim)
                # Dscaler.update()
                


                #Prep models
                self.modelG.train()
                for param in self.modelG.parameters():
                    param.grad  = None        
                for param in model_d.parameters():
                    param.grad  = None        



                #Autocast fake training
                with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=True):

                    #####################################################################
                    #                           TRAIN GENR                              #
                    #####################################################################

                    #Generate real labels
                    real_labels                                     = torch.ones(size=real_labels.shape,dtype=torch.float,device=self.device)

                    #Generate fake audio
                    generator_outputs                               = self.modelG(random_inputs)

                    #Classify fake audio
                    fake_classd                                     = model_d.forward(generator_outputs)
                    g_error                                         = self.error_fn(fake_classd,real_labels)    #Squared error??

                #Update parameters
                Gscaler.scale(g_error).backward()
                Gscaler.unscale_(self.G_optim)
                torch.nn.utils.clip_grad_norm_(modelG.parameters(), self.max_norm_g)
                Gscaler.step(self.G_optim)
                Gscaler.update()


                # #Autocast sim training
                # with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=True):
                #     #####################################################################
                #     #                          TRAIN SIMLIAR                            #
                #     #####################################################################

                #     rand_for_train          = self.modelG.generate_rand(bs=2)
                #     generator_outputs       = self.modelG(rand_for_train)           
                #     sim_err                 = self.difference_err(rand_for_train[0],rand_for_train[1],generator_outputs[0],generator_outputs[1])

                # #Update parameters
                # Gscaler.scale(sim_err).backward()
                # Gscaler.unscale_(self.G_optim)
                # torch.nn.utils.clip_grad_norm_(modelG.parameters(), self.max_norm_g)
                # Gscaler.step(self.G_optim)
                # Gscaler.update()

            printed = 0
            self.stats_handler.add_real(model_d.key,error=d_error_real.mean().detach().cpu().float().item())
            self.stats_handler.add_real(model_d.key,classification=torch.sigmoid(real_class.mean().detach().cpu().float()).item())

            self.stats_handler.add_fakeD(model_d.key,error=d_error_fake.mean().detach().cpu().float().item())
            self.stats_handler.add_fakeD(model_d.key,classification=torch.sigmoid(fake_class.mean().detach().cpu().float()).item())

            self.stats_handler.add_fakeG(model_d.key,g_error.mean().detach().cpu().float().item())
            self.stats_handler.add_rand(model_d.key,classifications=.5)#torch.sigmoid(rand_class.mean().detach().cpu().float()).item())
            
            

        # if verbose:
        #     percent = 1
        #     while (printed / num_equals) < percent:
        #         print("-",end='',flush=True)
        #         printed+=1
        


        #TELEMETRY

        print(f"]")
        print("\n")

        for i,key in enumerate([d.key for d in self.d_models]):
            
            out_2 = f"D(real)={self.stats_handler.get_d_real(key,mode='batch'):.4f}    D(gen)={self.stats_handler.get_d_fake(key,mode='batch'):.4f}    D(rand)={self.stats_handler.get_rand(key,mode='batch'):.4f}"
            print(" "*(width-len(out_2)),end='')
            print(f"\nD Model {i}")
            
            print(" "*(width-len(out_2)),end='')
            print(out_2)

            out_3 = f"er_real={self.stats_handler.get_d_real_err(key,mode='batch'):.4f}      er_fke={self.stats_handler.get_d_fake_err(key,mode='batch'):.4f}      g_error={self.stats_handler.get_g_err(key,mode='batch'):.3f}"
            print(" "*(width-len(out_3)),end='')
            print(out_3)
            
            #out_4   = f"rand_vect min={torch.min(random_outputs[0]).item():.4f}          rand_vect max={torch.max(random_outputs[0]).item():.4f}"
            #print(" "*(width-len(out_4)),end='')
            #print(out_4)

        if (d_error_real < .0001) and ((d_error_fake) < .0001):
            return True
        
        self.stats_handler.step_ep()



    #Calc error mult 
    def calc_err(self,class_d,class_g):

        #if class_d high and class_g low 
        if class_d > .75 and class_g < .35:
            self.k_real     = 0
            self.k_fake     = 0
        else:
            self.k_real     = .5
            self.k_fake     = .5 


    #Get a sample from Generator
    def sample(self,out_file_path,sample_set=25,store_plot=True,store_file="plots",n_samples=3):
       

        #Search for best sample  
        with torch.no_grad():
            self.modelG.eval()
            for model in self.d_models:
                model.eval()

            for i in range(n_samples):
                all_scores      = [] 
                best_score      = -100000
                best_sample     = None 
                for _ in range(sample_set):
                    
                    #Create inputs  
                    inputs  = self.modelG.generate_rand(bs=1)

                    #Grab score
                    outputs     = self.modelG.forward(inputs)
                    score       = 0 
                    for model in self.d_models:
                        score       += model(outputs).detach().cpu().mean().item()
                    score       /= len(self.d_models)
                    score       = torch.sigmoid(torch.tensor(score)).item()

                    #Check if better was found 
                    if score > best_score:
                        best_score      = score 
                        best_sample     = outputs.view(1,-1)
                    
                    all_scores.append(score)
            

                #Bring up to 2 channels 
                #waveform     = rebuild(torch.movedim(best_sample,0,1),self.u,self.s,self.n_dims).squeeze()
                waveform        = best_sample[0].cpu()
                waveform        = torchaudio.transforms.Resample(self.pca_handler.sample_rate,2048)(waveform)
                
                #Rescale up 
                data_utils.save_waveform_wav(waveform,out_file_path+f"_{i}.wav",2048)

            self.modelG.train()
            for model in self.d_models:
                model.train()
            self.plots.append(sorted(all_scores))
        #Telemetry 
        print(f"\tSAMPLE\n\t\tsample size: {sample_set}\n\t\tavg:\t{sum(all_scores)/len(all_scores):.3f}\n\t\tbest:\t{best_score:.3f}")

        #Store plot 
        if store_plot:
            plt.cla()
            all_scores = sorted(all_scores)

            #Plot current, 3 ago, 10 ago 
            plt.plot(list(range(sample_set)),all_scores,color="dodgerblue",label="current_p")
            try:
                if self.epoch_num >= 3:
                    plt.plot(list(range(sample_set)),self.plots[self.epoch_num-3],color="darkorange",label="prev-3_p")
                if self.epoch_num >= 10:
                    plt.plot(list(range(sample_set)),self.plots[self.epoch_num-10],color="crimson",label="prev-10_p")
            except IndexError:
                pass 
            plt.legend()
            plt.savefig(store_file)
            plt.cla()
            
            plt.ylim(-.1,.1)
            input_layer     = data_utils.reduce_arr(list(self.modelG.parameters())[-2].clone().flatten().cpu().detach().numpy(),500)
            input_layer.sort()
            plt.scatter(list(range(len(input_layer))),input_layer,color="crimson",label='layer[-1] weights')
            plt.legend()
            plt.savefig(store_file.replace('distros','weights'))
            plt.cla()
        

    #Train easier
    def c_exec(self,series_path:str,epochs:int=50,bs:int=8,verbose:bool=False):


        self.d_err_r_batch = []
        self.d_err_f_batch = []
        self.g_err_batches = []

        epochs = self.epoch_num+epochs
        
        self.build_dataset(bs,True,4)

        torch.backends.cudnn.benchmark = True

        #Load previous model
        if self.saved_state:
            self.epoch_num  = self.saved_state['epoch']
            self.modelG.state_dict   = self.load_models(self.saved_state["path"]+"/models/D_SAVE",self.saved_state["path"]+"/models/G_SAVE")


        for e in range(self.epoch_num,epochs):
            self.epoch_num      = e 
            t0 = time.time()
            
            if verbose:
                print_epoch_header(e,epochs)
            self.train_mp(verbose=verbose)#,gen_train_iters=gen_train_iters,proto_optimizers=proto_optimizers)
            #self.train_wasserstein(verbose=verbose,t_dload=time.time()-t0,proto_optimizers=False)#,gen_train_iters=gen_train_iters,proto_optimizers=proto_optimizers)

            #if (e+1) % sample_rate == 0:
            self.save_run(series_path)
            #random.shuffle(self.pca_handler.data_vectors)

            #Adjust learning rates 
            if self.lr_adjust and False:
                slope       = linregress(range(len(self.g_err_batches)),self.g_err_batches)[0]

                
                if e < 114:
                    self.D_optim.param_groups[0]['lr']      *= .98
                print(f"\n\tHYPERPARAMS")
                print(f"\t\tg_err slope:\t{slope:.6f}")

                if slope > 0 and self.training_classes['g_class'][-1] < .4 or self.training_classes["g_class"][-1] < .175:
                    self.G_optim.param_groups[0]['lr']	*= 1.08
                    incr        = "↑" 

                elif len(self.training_classes['g_class']) > 2 and self.training_classes['g_class'][-2] < self.training_classes['g_class'][-1]  or self.training_classes['g_class'][-1] > .49:
                    self.G_optim.param_groups[0]['lr']	*= .94
                    incr        = "↓"
                else:
                    incr        = "↔"
                print(f"\t\tD_lr:\t\t{self.D_optim.param_groups[0]['lr']:.6f}\n\t\tG_lr:\t\t{self.G_optim.param_groups[0]['lr']:.6f} ({incr})")
                print(f"\n\n")


    def set_colors(self):
        self.g_err_colors        = ['forestgreen','springgreen','lawngreen','seagreen']
        self.d_err_real_colors   = ['goldenrod','darkorange','gold','tan']
        self.d_err_fake_colors   = ['black','dimgrey','silver','darkgrey']

        self.real_class_colors   = ['darkorchid','rebeccapurple','violet','fuchsia']
        self.fake_class_colors   = ['forestgreen','springgreen','lawngreen','seagreen']
        self.rand_class_colors   = ['darkkhaki','wheat','yellow','orange']


    #Save telemetry and save to file 
    def save_run(self,series_path,sample_set=100):
        
        #Create samples && plots 
        self.sample(os.path.join(series_path,'samples',f'run{self.epoch_num}'),store_file=os.path.join(series_path,'distros',f'run{self.epoch_num}'),sample_set=sample_set)

        #Create errors and classifications 
        plt.cla()
        fig,axs     = plt.subplots(nrows=2,ncols=1)
        fig.set_size_inches(30,16)




        for i,d in enumerate(self.d_models):
            axs[0].plot(list(range(250)),data_utils.reduce_arr(self.stats_handler.get_g_err(key=d.key,returntype='list'),250),label=f"G_err {i}",color=self.g_err_colors[i])
            axs[0].plot(list(range(250)),data_utils.reduce_arr(self.stats_handler.get_d_real_err(key=d.key,returntype='list'),250),label=f"D_err_real {i}",color=self.d_err_real_colors[i])
            axs[0].plot(list(range(250)),data_utils.reduce_arr(self.stats_handler.get_d_fake_err(key=d.key,returntype='list'),250),label=f"D_err_fake {i}",color=self.d_err_fake_colors[i])

            axs[1].plot(list(range(250)),data_utils.reduce_arr(self.stats_handler.get_d_real(key=d.key,returntype='list'),250),label=f"Real Class {i}",color= self.real_class_colors[i])
            axs[1].plot(list(range(250)),data_utils.reduce_arr(self.stats_handler.get_d_fake(key=d.key,returntype='list'),250),label=f"Fake Class {i}",color= self.fake_class_colors[i])
            axs[1].plot(list(range(250)),data_utils.reduce_arr(self.stats_handler.get_rand(key=d.key,returntype='list'),250),label=f"Rand Class {i}",color=self.rand_class_colors[i])


        axs[0].set_title("Model Loss per Batch")
        axs[0].set_xlabel("Batch #")
        axs[0].set_ylabel("BCE Loss")
        axs[0].legend()

        axs[1].set_title("Model Classifications per Batch")
        axs[1].set_xlabel("Batch  #")
        axs[1].set_ylabel("Classification")
        axs[1].legend()


        fig.savefig(f"{os.path.join(series_path,'errors',f'Error and Classifications')}")
        plt.close()
        #Save models 
        self.save_model_states(os.path.join(series_path,"models"))
        self.stats_handler.epoch_num    = self.epoch_num
        self.stats_handler.save_state(series_path)

        #Save telemetry to file every step - will try to be recovered at beginning
        # stash_dictionary    = json.dumps({"training_errors":self.training_errors,"training_classes":self.training_classes,"epoch":self.epoch_num,"path":series_path})
        # with open(os.path.join(series_path,"data","data.txt"),"w") as save_file:
        #     save_file.write(stash_dictionary)
        # save_file.close()


    def load_run(self,series_path):

        self.stats_handler.load_state(series_path)
        self.epoch_num  = self.stats_handler.epoch_num

        self.load_models(series_path)


    def difference_err(self,z1,z2,out1,out2):

        z_sim           = torch.dot(z1[:,0],z2[:,0]) / (torch.norm(z1[:,0])*torch.norm(z2[:,0]))
        out_sim         = torch.dot(out1[0],out2[0]) / (torch.norm(out1[0])*torch.norm(out2[0]))

        return          torch.abs(z_sim-out_sim)

        #get layer 1 weights
        

#Training Section
if __name__ == "__main__":
    dev         = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    ep          = eval(sys.argv[sys.argv.index("ep")+1]) if "ep" in sys.argv else 50
    n_z         = eval(sys.argv[sys.argv.index("nz")+1]) if "nz" in sys.argv else 256
    rank        = eval(sys.argv[sys.argv.index("rank")+1]) if "rank" in sys.argv else 4096+1024
    bs          = eval(sys.argv[sys.argv.index("bs")+1]) if "bs" in sys.argv else 32
    n_block     = 4

    print(f"running\n\tbs\t{bs}\n\tn_z\t{n_z}")
    if not os.path.exists("model_runs"):
        os.mkdir(f"model_runs")


    configs = {"Upsampler"   : {"model":networks.WaveGen2,"model_args":{},"lrs": (5e-5,1e-4), "n_z":n_z,"momentum":.9,"bs":bs,"wd":1e-2,"iters":250,"pca_rank":rank,"act_fn":torch.nn.ReLU}}

    for config in configs:  

        #Load values 
        name                    = config 
        classloader             = configs[config]['model']
        args                    = configs[config]['model_args']
        n_z                     = configs[config]['n_z'] 
        lrs                     = configs[config]['lrs']
        momentum                = configs[config]['momentum']
        bs                      = configs[config]['bs']
        wd                      = configs[config]['wd']
        pca_rank                = configs[config]['pca_rank']
        act_fn                  = configs[config]['act_fn']
        series_path             = f"model_runs/{config}_bs={bs}_nz={n_z}"
        loading                 = False if not "lf" in sys.argv else eval(sys.argv[sys.argv.index("lf")+1]) 
        max_load                = 1_000_000
        sample_rate             = 1024

        #Create handler
        # pca_handler             = data_utils.PCA_Handler(from_wav_folder="C:/data/music/wav/")
        # pca_handler.construct_pca_from_wavs(sample_rate=2048,pca_rank=4096,n_samples=1024)
        pca_handler             = data_utils.PCA_Handler(from_vectors="C:/data/music/2048_2048_16/",sample_rate=sample_rate,non_pca=True)
        pca_handler.ds_no_pca(n_samples=1024)
        pca_rank                = pca_handler.pca_rank
        print(f"\tds_len\t{len(pca_handler)}\n\tpath\t{series_path}")

        #Build models 
        #modelG:SoundModel       = classloader(n_z=n_z,n_out=pca_rank,handler=pca_handler,**args,act_fn=act_fn,n_block=n_block)
        modelG:SoundModel       = classloader(n_z=n_z,sample_rate=sample_rate//2,upsample_rate=sample_rate,**args,act_fn=act_fn)
        #modelG.apply(data_utils.weights_init)
        #modelG.load_state_dict(torch.load("C:/gitrepos/lofi/model_runs/multiDStraight_bs=32_nz=100/models/G_SAVE"))
        #modelD2:SoundModel      = networks.GPTDiscriminator(32768,act_fn=torch.nn.ReLU,n_kernel=4,k1=33,k2=9,k3=5)
        modelD1:SoundModel      = networks.WAVDiscriminator(act_fn=torch.nn.LeakyReLU,n_kernel=16,out_size=sample_rate*16)


        d_models                = [modelD1]
        #Build Trainer 
        t:Trainer               = Trainer(modelG,d_models,pca_handler)
        t.saved_state           = False
        t.max_iters             = configs[config]['iters']
        t.max_norm_d            = 1
        t.max_norm_g            = 1
        t.bs                    = bs
        pca_handler.max_load    = min(len(pca_handler),max_load)

        #Create folder system 
        if not os.path.exists(series_path):
            os.mkdir(series_path)
            for folders in ["samples","errors","distros","models","data","weights"]:
                os.mkdir(os.path.join(series_path,folders))
        else:
            if not loading:
                print(f"\tep\t{t.epoch_num}")
                pass 
            else:
                t.load_run(series_path)
                print(f"\tep\t{t.epoch_num} (loaded)")


        t.override              = False
        t.lr_adjust             = True
        t.stop_thresh           = .9
        t.start_thresh          = .65
        t.stopped               = False
        t.warmup                = 0

        #Check output sizes are kosher 
        inpv2                   = modelG.generate_rand(bs=1)
        print(f"\nMODEL INFO")
        if not loading:
            t.epoch_num             = 0


        with torch.no_grad():
            print(f"\tG stats:\tout  - {modelG.forward(inpv2).shape }\n        \t\tsize - {modelG.size()/1000000:.2f}MB\n        \t\tparams: - {modelG.params()/1000000:.2f}M")

            for md in d_models:
                print(f"\tD stats:\tout  - {md( modelG.forward(inpv2)).shape}\n        \t\tsize - {md.size()/1000000:.2f}MB\n        \t\tparams: - {md.params()/1000000:.2f}M\n        \t\teval  - {md(modelG.forward(inpv2))[0].detach().cpu().item():.3f}")

        #Create optims 
        #optim_d = torch.optim.SGD(D.parameters(),lr=lrs[0],momentum=momentum)#lrs[0],betas=(momentum,.999))
        #optim_g = torch.optim.SGD(G.parameters(),lr=lrs[1],momentum=momentum)#lrs[1],betas=(momentum,.999))
    
        optimD1                 = torch.optim.Adam(modelD1.parameters(),lrs[0],betas=(.5,.999),weight_decay=0)#,momentum=momentum)
        #optimD2                 = torch.optim.Adam(modelD2.parameters(),lrs[0],betas=(.5,.999),weight_decay=0)#,momentum=momentum)
        optimG                  = torch.optim.Adam(modelG.parameters(),lrs[1],betas=(.5,.999),weight_decay=0)#,momentum=momentum)

        # t.outsize        = outsize
        # t.n_z            = n_z
        # t.set_learners(optim_d,optim_g,torch.nn.BCELoss())
        t.set_learners([optimD1],optimG,error_fn=torch.nn.BCEWithLogitsLoss())
        t.c_exec(series_path,epochs=ep,bs=bs,verbose=True)
        print(f"done")

        #t.train_gen_on_real(filenames=files,bs=16,series_path=series_path)