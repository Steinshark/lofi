import torch 
from collections import OrderedDict
from networks import *
from torch.utils.data import Dataset, DataLoader
import numpy 
import time 
import json
import os 
from data_utils import *
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
from networks import Generator,SoundModel,Discriminator
import data_utils


#Provides the mechanism to train the model out
class Trainer:

    #Initialize
    def __init__(self,modelG:SoundModel,modelD:SoundModel,pca_handler:PCA_Handler):

        #Prepare models
        self.modelG             = modelG 
        self.modelD             = modelD 
        
        self.pca_handler        = pca_handler

        self.device             = self.modelG.device
        self.n_z                = self.modelG.n_z


        self.plots              = []
        self.training_errors    = {     "g_err":     [],
                                        "d_err_real":[],
                                        "d_err_fake":[]}

        self.training_classes   = {     "g_class":[],
                                        "d_class":[],
                                        "r_class":[]}
    
    #Import a list of configs for D and G in JSON format.
    def import_configs(self,config_filename:str):

        #Check if file exists
        if not os.path.exists(config_filename):
            print(f"Config file {config_filename} does not exist")
            return

        #Open file and read to JSON 
        file_contents       = open(config_filename,"r").read()
        try:
            config_dictionary   = json.loads(file_contents)

            #Ensure proper format 
            if not "G" in config_dictionary or not "D" in config_dictionary:
                print("Incorrectly formatted config file: Must contain G and D entries")
                return 
            
            self.config_file = config_dictionary

        except json.JSONDecodeError:
            print(f"file not in correct JSON format")
            return 

    #Create models from scratch using a config
    def create_models(self,D_config:dict,G_config:dict,run_parallel=True):

        #Ensure Proper Config Files 
        if not "factors" in G_config:
            print("Invalid Generator config, must contain Factor settings")
            exit(-1) 
        if not "channels" in G_config:
            print("Invalid Generator config, must contain Channel settings")
            exit(-1) 
        if not "scales" in G_config:
            print("Invalid Generator config, must contain Scale settings")
            exit(-1) 
       
        if not "kernels" in D_config:
            print("Invalid Discrimintator config, must contain Kernel settings")
            exit(-1) 
        if not "strides" in D_config:
            print("Invalid Discrimintator config, must contain Stride settings")
            exit(-1) 
        if not "paddings" in D_config:
            print("Invalid Discrimintator config, must contain Padding settings")
            exit(-1) 
        if not "channels" in D_config:
            print("Invalid Discrimintator config, must contain Channels settings")
            exit(-1)
        

        #Create Generator 
        self.Generator   = AudioGenerator2(         G_config['factors'],
                                                    G_config['channels'],
                                                    G_config['scales'])

        #Create Discriminator
        self.Discriminator   = AudioDiscriminator(  channels=D_config['channels'],
                                                    kernels=D_config['kernels'],
                                                    strides=D_config['strides'],
                                                    paddings=D_config['paddings'],
                                                    final_layer=D_config['final_layer'],
                                                    device=self.device,
                                                    verbose=False)

        #Init weights 
        self.Generator.apply(weights_initD)
        self.Discriminator.apply(weights_initD)

        #Check if mulitple GPUs 
        if torch.cuda.device_count() > 1 and run_parallel:
            print(f"Running model on {torch.cuda.device_count()} distributed GPUs")
            self.Generator      = torch.nn.DataParallel(self.Generator,device_ids=[id for id in range(torch.cuda.device_count())])
            self.Discriminator  = torch.nn.DataParallel(self.Discriminator,device_ids=[id for id in range(torch.cuda.device_count())])

        #Put both models on correct device 
        self.Generator      = self.Generator.to(self.device)
        self.Discriminator  = self.Discriminator.to(self.device) 
        #Save model config for later 
        self.G_config = G_config
        self.D_config = D_config

    #Import saved models 
    def load_models(self,D_params_fname:str,G_params_fname:str,ep_start=None):
        
        #Create models
        #self.create_models(D_config,G_config)

        #Load params
        try:
            self.Generator.load_state_dict(     torch.load(G_params_fname))
            self.Discriminator.load_state_dict( torch.load(D_params_fname))
            print(f"loaded for epoch {self.epoch_num}")
        except FileNotFoundError:
            return

    #Save state dicts and model configs
    def save_model_states(self,path:str,D_name="Discriminator_1",G_name="Generator_1"):
        
        #Ensure path is good 
        if not os.path.exists(path):
            os.mkdir(path)

        #Save model params to file 
        torch.save(     self.Generator.state_dict(),        os.path.join(path,G_name))
        torch.save(     self.Discriminator.state_dict(),    os.path.join(path,D_name)) 
        
        ##Save configs
        #with open(os.path.join(path,f"{G_name}_config"),"w") as config_file:
        #    config_file.write(json.dumps(self.G_config))
        #    config_file.close()

        #with open(os.path.join(path,f"{D_name}_config"),"w") as config_file:
        #    config_file.write(json.dumps(self.D_config))
        #    config_file.close()

    #Get incoming data into a dataloader
    def build_dataset(self,batch_size:int,shuffle:bool,num_workers:int):

        #Save parameters 
        self.batch_size     = batch_size

        #Create sets
        self.dataset        = AudioDataSet(self.pca_handler)
        self.dataloader     = DataLoader(self.dataset,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers,pin_memory=False)

    #Set optimizers and error function for models
    def set_learners(self,D_optim,G_optim,error_fn):
        self.D_optim    = D_optim
        self.G_optim    = G_optim
        self.error_fn   = error_fn

    #Train Vanilla
    def train(self,verbose=True,gen_train_iters=1,proto_optimizers=True,t_dload=0,train_rand=False,load=1_000_000):
        t_start = time.time()
        torch.backends.cudnn.benchmark = True


        #Telemetry
        if verbose:
            width       = 100
            num_equals  = 50
            indent      = 4 
            n_batches   = min(len(self.dataset) / self.batch_size,load/self.batch_size) 
            t_init      = time.time()
            printed     = 0
            t_d         = [0] 
            t_g         = [0] 
            t_op_d      = [0]
            t_op_g      = [0]    
            d_fake      = 0 
            d_fake2     = 0 
            d_real      = 0
            d_random    = 0 

            print(" "*indent,end='')
            prefix = f"{int(n_batches)} batches  Progress" 
            print(f"{prefix}",end='')
            print(" "*(width-indent-len(prefix)-num_equals-2),end='')
            print("[",end='')
        
        d_error_fake = 0 
        d_error_real = 0 
        g_error = 0

        d_err_r_b   = [] 
        d_err_f_b   = [] 
        g_err       = [] 

        d_class_r   = [] 
        d_class_g   = [] 

        

        #Run all batches
        for i, data in enumerate(self.dataloader,0):

            #Keep track of which batch we are on 
            final_batch     = (i == len(self.dataloader)-1) or (i*self.batch_size >= load)

            if verbose:
                percent = i / n_batches
                while (printed / num_equals) < percent:
                    print("-",end='',flush=True)
                    printed+=1

            #####################################################################
            #                           TRAIN REAL                              #
            #####################################################################
            
            #Zero First
            # OLD IMPLEMENTATION: self.Discriminator.zero_grad()
            for param in self.Discriminator.parameters():
                param.grad = None

            #Prep real values
            t0                  = time.time()
            x_set               = data[0].to(self.device)
            x_len               = len(x_set)
            y_set               = torch.ones(size=(x_len,),dtype=torch.float,device=self.device)
            data_l              = time.time() - t0
            

            #Classify real set
            t_0 = time.time()
            real_class          = self.Discriminator.forward(x_set).view(-1)
            classification_d    = real_class.mean().item()
            d_real              += classification_d
            self.training_classes['d_class'].append(classification_d)
            d_class_r.append(classification_d)

            t_d[-1]             += time.time()-t_0

            #Calc error
            t0                  = time.time()
            d_error_real        = self.error_fn(real_class,y_set)
            d_err_r_b.append(d_error_real.mean().cpu().detach().float())
            d_error_real.backward()
            t_op_d[-1]          += time.time() - t0
            
            #####################################################################
            #                           TRAIN FAKE                              #
            #####################################################################
            
            #Generate samples
            t_0 = time.time()
            random_inputs           = self.modelG.generate_rand(bs=self.batch_size)
            generator_outputs       = self.Generator(random_inputs)
            t_g[-1] += time.time()-t_0

            #Ask Discriminator to classify fake samples 
            t_0 = time.time()
            fake_labels             = torch.zeros(size=(x_len,),dtype=torch.float,device=self.device)
            fake_class              = self.Discriminator.forward(generator_outputs.detach()).view(-1)
            classification_g        = fake_class.mean().item()
            d_fake                  += classification_g
            self.training_classes['g_class'].append(classification_g)
            d_class_g.append(classification_g)

            t_d[-1] += time.time()-t_0

            #Calc error
            t_0 = time.time()
            d_error_fake            = self.error_fn(fake_class,fake_labels)
            d_err_f_b.append(d_error_fake.mean().cpu().float().detach()) 
            d_error_fake.backward()


            #Check for restart
            if self.stopped and classification_d < self.start_thresh or self.override:
                self.stopped    = False 
                self.D_optim.step()   
                t_op_d[-1] += time.time()-t_0

            #Check for stop
            elif classification_d > self.stop_thresh:
                self.stopped    = True
            
            #Check for stopped
            elif self.stopped:
                continue
            
            #Check for normal
            else:
                self.D_optim.step()   
                t_op_d[-1] += time.time()-t_0




           
            #####################################################################
            #                           TRAIN GENR                              #
            #####################################################################

            #Zero Grads
            for param in self.Generator.parameters():
                param.grad = None

            #Classify the fakes again after Discriminator got updated 
            t_0 = time.time()
            fake_class2                 = self.Discriminator.forward(generator_outputs).view(-1)
            g_class                     = fake_class2.mean().item()
            t_d[-1] += time.time()-t_0
            
            #Find the error between the fake batch and real set  
            t_0 = time.time()
            g_error                 = self.error_fn(fake_class2,y_set)
            g_err.append(g_error.mean().cpu().float().detach()) 
            t_op_g[-1] += time.time()-t_0
            
            #Back Propogate
            t_0 = time.time()
            g_error.backward()   
            if classification_g < self.stop_thresh or self.override:
                self.G_optim.step()
                t_op_g[-1] += time.time()-t_0


            #####################################################################
            #                            GET RAND                               #
            #####################################################################
            with torch.no_grad():
                random_vect             = torch.randn(size=(1,self.outsize[0],self.outsize[1]),dtype=torch.float,device=self.device)
                d_random                = self.Discriminator.forward(random_vect).cpu().detach()
            for tensor in d_random:
                self.training_classes['r_class'].append(tensor.item())
            
            


        if verbose:
            percent = (i+1) / n_batches
            while (printed / num_equals) < percent:
                print("-",end='',flush=True)
                printed+=1
        


        #TELEMETRY
        print(f"]")
        print("\n")
        out_1 = f"D forw={sum(t_d):.3f}s    G forw={sum(t_g):.3f}s    D back={sum(t_op_d):.3f}s    G back={sum(t_op_g):.3f}s    tot = {(time.time()-t_init):.2f}s"
        print(" "*(width-len(out_1)),end='')
        print(out_1,flush=True)
        out_2 = f"t_dload={(t_dload):.2f}s    D(real)={(d_real/n_batches):.3f}    D(gen1)={(d_fake/n_batches):.4f}    D(rand)={d_random.item():.3f}"

        print(" "*(width-len(out_2)),end='')
        print(out_2)
        
        out_3 = f"er_real={(d_error_real):.3f}     er_fke={(d_error_fake):.4f}    g_error={(g_error):.3f}"
        print(" "*(width-len(out_3)),end='')
        print(out_3)
       
        t_d.append(0)
        t_g.append(0)

        self.training_errors['d_err_real'].append(d_error_real.cpu().item())
        self.training_errors['d_err_fake'].append(d_error_fake.cpu().item())
        self.training_errors['g_err']     .append(g_error.cpu().item())

        #self.training_classes['r_class'].append(d_random)

        self.d_err_r_batch      += d_err_r_b
        self.d_err_f_batch      += d_err_f_b
        self.g_err_batches      += g_err 

        if (d_error_real < .0001) and ((d_error_fake) < .0001):
            return True


    def train_wasserstein(self,verbose=True,gen_train_iters=1,proto_optimizers=True,t_dload=0,train_rand=False):
        t_start = time.time()
        torch.backends.cudnn.benchmark = True


        #Telemetry
        if verbose:
            width       = 100
            num_equals  = 50
            indent      = 4 
            n_batches   = len(self.dataset) / self.batch_size 
            t_init      = time.time()
            printed     = 0
            d_err       = 0 
            g_err       = 0 

            print(" "*indent,end='')
            prefix = f"{int(n_batches)} batches  Progress" 
            print(f"{prefix}",end='')
            print(" "*(width-indent-len(prefix)-num_equals-2),end='')
            print("[",end='')
        
        d_errs          = []
        g_errs          = []
        

        #Run all batches
        for i, data in enumerate(self.dataloader,0):

            if verbose:
                percent = i / n_batches
                while (printed / num_equals) < percent:
                    print("-",end='',flush=True)
                    printed+=1

            #Prep real values
            x_set               = data[0].to(self.device)
            x_len               = len(x_set)
            y_set               = torch.ones(size=(x_len,),dtype=torch.float,device=self.device)


            #####################################################################
            #                           TRAIN CRIT                              #
            #####################################################################
            for _ in range(1):
                

                #Classify real set
                real_class          = self.Discriminator.forward(x_set).view(-1)
                random_inputs       = torch.randn(size=(x_len,self.n_z,1),dtype=torch.float,device=self.device)
                generated_samples   = self.Generator.forward(random_inputs)
                fake_class          = self.Discriminator.forward(generated_samples).reshape(-1)
                grad_penalty        = self.gradient_penalty(self.Discriminator,x_set,generated_samples)

                loss_d              = -(torch.mean(real_class) - torch.mean(fake_class)) + lambda_gp * grad_penalty
        
                self.Discriminator.zero_grad()
                self.Generator.zero_grad()

                loss_d.backward(retain_graph=True)
                self.D_optim.step()

            #####################################################################
            #                           TRAIN GENR                              #
            #####################################################################
            
            #Generate samples
            generator_outputs       = self.Discriminator(generated_samples).reshape(-1)
            loss_g                  = -torch.mean(generator_outputs)

            self.Generator.zero_grad()

            loss_g.backward()
            self.G_optim.step()

            d_err += float(loss_d.item())
            g_err += float(loss_g.item())

            d_errs.append(loss_d.item())
            g_errs.append(loss_g.item())

        if verbose:
            percent = (i+1) / n_batches
            while (printed / num_equals) < percent:
                print("-",end='',flush=True)
                printed+=1


        print(f"]\n\n\t\tEpoch {self.epoch_num+1}\tlossD {(d_err/n_batches):.4f}\tlossG:{(g_err/n_batches):.4f}\n")

        self.training_errors['d_err_real'].append(d_err)
        self.training_errors['d_err_fake'].append(d_err)
        self.training_errors['g_err']     .append(g_err)

        self.training_classes['d_class'].append(real_class.mean().cpu().item())
        self.training_classes['g_class'].append(fake_class.mean().cpu().item())
        self.training_classes['r_class'].append(0)

        self.d_err_r_batch      += d_errs
        self.d_err_f_batch      += [0] * len(g_errs)
        self.g_err_batches      += g_errs 


    def gradient_penalty(self,critic, real, fake):
        epsilon = torch.rand(self.batch_size,1,1).to(self.device)
        interpolated_images = real*epsilon + fake*(1-epsilon)
        
        # Calculate critic scores
        mixed_scores = critic(interpolated_images)
        
        gradient = torch.autograd.grad(
            inputs = interpolated_images,
            outputs = mixed_scores,
            grad_outputs = torch.ones_like(mixed_scores),
            create_graph = True,
            retain_graph = True,
        )[0]
        
        gradient = gradient.view(gradient.shape[0], -1)
        gradient_norm = gradient.norm(2, dim = 1)
        gradient_penalty = torch.mean((gradient_norm - 1)**2)
        return gradient_penalty
    
    #Get a sample from Generator
    def sample(self,out_file_path,sf=1,sample_set=100,store_plot=True,store_file="plots",n_samples=3):
       

        #Search for best sample  
        with torch.no_grad():
            for i in range(n_samples):
                all_scores      = [] 
                best_score      = -100000
                best_sample     = None 
                for _ in range(sample_set):
                    
                    #Create inputs  
                    if self.mode == "multi-channel":
                        inputs  = torch.randn(size=(1,self.n_z,1),dtype=torch.float,device=self.device)
                    elif self.mode == "single-channel":
                        inputs  = torch.randn(size=(1,1,self.n_z),dtype=torch.float,device=self.device)
                    
                    #Grab score
                    outputs     = self.Generator.forward(inputs)
                    score       = self.Discriminator(outputs).mean().item()

                    #Check if better was found 
                    if score > best_score:
                        best_score      = score 
                        best_sample     = outputs.view(1,-1)
                    
                    all_scores.append(score)
            

                #Bring up to 2 channels 
                #waveform     = rebuild(torch.movedim(best_sample,0,1),self.u,self.s,self.n_dims).squeeze()
                print(f"waveform shape {best_sample.shape}")
                waveform        = best_sample[0].cpu()
                
                #Rescale up 
                tensor_to_wav(waveform,out_file_path+f"_{i}.wav",1024)
                #torchaudio.save(out_file_path+f"_{i}.wav",waveform.repeat(2,1),1024)
            self.plots.append(sorted(all_scores))
        #Telemetry 
        print(f"\tSAMPLE\n\t\tsample size: {sample_set}\n\t\tavg:\t{sum(all_scores)/len(all_scores):.3f}\n\t\tbest:\t{best_score:.3f}")

        #Store plot 
        if store_plot:
            plt.cla()
            all_scores = sorted(all_scores)

            #Plot current, 3 ago, 10 ago 
            plt.plot(list(range(sample_set)),all_scores,color="dodgerblue",label="current")
            try:
                if self.epoch_num >= 3:
                    plt.plot(list(range(sample_set)),self.plots[self.epoch_num-3],color="darkorange",label="prev-3")
                if self.epoch_num >= 10:
                    plt.plot(list(range(sample_set)),self.plots[self.epoch_num-10],color="crimson",label="prev-10")
            except IndexError:
                pass 
            plt.legend()
            plt.savefig(store_file)
            plt.cla()

        

    #Train easier
    def c_exec(self,load,epochs,bs,optim_d,optim_g,n_z,outsize,filenames,series_path,sf,verbose=False,sample_rate=1,rebuild_dataset=False):
        self.outsize        = outsize
        self.n_z            = n_z
        self.sf             = sf


        self.d_err_r_batch = []
        self.d_err_f_batch = []
        self.g_err_batches = []

        self.set_learners(optim_d,optim_g,torch.nn.BCELoss())   
        epochs = self.epoch_num+epochs
        

        self.build_dataset(bs,True,4)

        #Load previous model
        if self.saved_state:
            self.epoch_num  = self.saved_state['epoch']
            self.Generator.state_dict   = self.load_models(self.saved_state["path"]+"/models/D_SAVE",self.saved_state["path"]+"/models/G_SAVE")


        for e in range(self.epoch_num,epochs):
            self.epoch_num      = e 
            t0 = time.time()
            
            if verbose:
                print_epoch_header(e,epochs)
            failed = self.train(verbose=verbose,t_dload=time.time()-t0,proto_optimizers=False,load=load)#,gen_train_iters=gen_train_iters,proto_optimizers=proto_optimizers)
            #failed = self.train_wasserstein(verbose=verbose,t_dload=time.time()-t0,proto_optimizers=False)#,gen_train_iters=gen_train_iters,proto_optimizers=proto_optimizers)

            #if (e+1) % sample_rate == 0:
            self.save_run(series_path)
            if failed:
                return 
            self.save_model_states("models","D_model","G_model")
            

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

    #Save telemetry and save to file 
    def save_run(self,series_path,sample_set=100):
        
        #Create samples && plots 
        self.sample(os.path.join(series_path,'samples',f'run{self.epoch_num}'),store_file=os.path.join(series_path,'distros',f'run{self.epoch_num}'),sample_set=sample_set)

        #Create errors and classifications 
        plt.cla()
        fig,axs     = plt.subplots(nrows=3,ncols=1)
        fig.set_size_inches(30,16)
        axs[0].plot(list(range(250)),reduce_arr(self.training_errors['g_err'],250),label="G_err",color="goldenrod")
        axs[0].plot(list(range(250)),reduce_arr(self.training_errors['d_err_real'],250),label="D_err_real",color="darkorange")
        axs[0].plot(list(range(250)),reduce_arr(self.training_errors['d_err_fake'],250),label="D_err_fake",color="dodgerblue")
        axs[0].set_title("Model Loss vs Epoch")
        axs[0].set_xlabel("Epoch #")
        axs[0].set_ylabel("BCE Loss")
        axs[0].legend()

        axs[1].plot(list(range(250)),reduce_arr(self.training_classes['d_class'],250),label="Real Class",color="darkorange")
        axs[1].plot(list(range(250)),reduce_arr(self.training_classes['g_class'],250),label="Fake Class",color="dodgerblue")
        axs[1].plot(list(range(250)),reduce_arr(self.training_classes['r_class'],250),label="Rand Class",color="dimgrey")
        axs[1].set_title("Model Classifications vs Epoch")
        axs[1].set_xlabel("Epoch #")
        axs[1].set_ylabel("Classification")
        axs[1].legend()

        axs[2].plot(list(range(250)),reduce_arr(self.d_err_r_batch,250),label="D_err Real",color="darkorange")
        axs[2].plot(list(range(250)),reduce_arr(self.d_err_f_batch,250),label="D_err Fake",color="dodgerblue")
        axs[2].plot(list(range(250)),reduce_arr(self.g_err_batches,250),label="G_err",color="goldenrod")
        axs[2].set_title("Model Errors per Batch")
        axs[2].set_xlabel("Batch #")
        axs[2].set_ylabel("BCE Error")
        axs[2].legend()

        fig.savefig(f"{os.path.join(series_path,'errors',f'Error and Classifications')}")
        plt.close()
        #Save models 
        self.save_model_states(os.path.join(series_path,'models'),D_name=f"D_SAVE",G_name=f"G_SAVE")

        #Save telemetry to file every step - will try to be recovered at beginning
        stash_dictionary    = json.dumps({"training_errors":self.training_errors,"training_classes":self.training_classes,"epoch":self.epoch_num,"path":series_path})
        with open(os.path.join(series_path,"data","data.txt"),"w") as save_file:
            save_file.write(stash_dictionary)
        save_file.close()

    #Start with pretrained Generator
    def train_gen_on_real(self,filenames,bs,series_path=""):
        
        self.d_err_r_batch = []
        self.d_err_f_batch = []
        self.g_err_batches = []

        self.random_matches = {}

        optim_g = torch.optim.Adam(self.Generator.parameters(),lr=.00002,weight_decay=.000002,betas=(.5,.99))
        self.set_learners(optim_d,optim_g,torch.nn.BCELoss())
        train_set           = random.sample(filenames,load)
        self.build_dataset(train_set,load,bs,True,4)


        loss_fn             = torch.nn.MSELoss()

        self.epoch_num      = 0
        self.save_run(series_path)



        for e in range(50):
            for b, batch in enumerate(self.dataloader):

                for p in self.Generator.parameters():
                    p.grad              = None 

                data                = batch[0].to(torch.device('cuda'))
                names               = batch[1]

                rand_in             = torch.zeros(size=(bs,n_z,1),dtype=torch.float,device=self.device)

                for i,name in enumerate(names):
                    if name in self.random_matches:
                        rand_in[i]                  = self.random_matches[name].clone()
                    else:
                        self.random_matches[name]   = torch.randn(size=(1,n_z,1),dtype=torch.float,device=self.device)
                        rand_in[i]                  = self.random_matches[name]
                
                bs                  = len(data)

                gen_outs            = self.Generator.forward(rand_in)

                
                loss                = loss_fn(data,gen_outs)
                loss_val            = loss.mean().float()
                loss.backward()
                self.G_optim.step()
                if b % 64 == 0:
                    print(f"\t{b}/{len(self.dataloader)}\tloss was {loss_val:.5f}")
            self.epoch_num += 1
            self.save_run(series_path)

        self.save_model_states(os.path.join(series_path,'models'),D_name=f"D_SAVE",G_name=f"G_SAVE")


#Training Section
if __name__ == "__main__":
    dev         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ep          = eval(sys.argv[sys.argv.index("ep")+1]) if "ep" in sys.argv else 50
    load        = eval(sys.argv[sys.argv.index("ld")+1]) if "ld" in sys.argv else 2048
    outsize     = (1,eval(sys.argv[sys.argv.index("out")+1])) if "out" in sys.argv else (1,int(4096))
    n_z         = eval(sys.argv[sys.argv.index("nz")+1]) if "n_z" in sys.argv else 128

    if not os.path.exists("model_runs"):
        os.mkdir(f"model_runs")


    pca_handler         = PCA_Handler(from_wav_folder="C:/data/music/wav/")
    pca_handler.construct_pca_from_wavs(sample_rate=2048,pca_rank=2048+1024,n_samples=200)

    configs = {"svd_fcn1"   : {"model":Generator,"model_args":{"n_z":200,"n_out":2048+1024,"dropout":.1,'handler':pca_handler},"lrs": (.5e-4,.5e-5), "n_z":250,"momentum":.5,"bs":4,"wd":2e-6,"iters":250}}

    for config in configs:  

        #Load values 
        name                    = config 
        classloader             = configs[config]['model']
        args                    = configs[config]['args']
        n_z                     = configs[config]['n_z'] 
        lrs                     = configs[config]['lrs']
        momentum                = configs[config]['momentum']
        leak                    = configs[config]['dropout']
        bs                      = configs[config]['bs']
        wd                      = configs[config]['wd']
        series_path             = f"model_runs/{config}_{bs}_{n_z}"
        loading                 = False if not "lf" in sys.argv else eval(sys.argv[sys.argv.index("lf")+1]) 
        lambda_gp               = 10

        #Build models 
        modelG:SoundModel       = classloader(**args)
        modelD:SoundModel       = Discriminator()

        #Build Trainer 
        t                       = Trainer(modelG,modelD,pca_handler)
        t.saved_state           = False
        t.max_iters             = configs[config]['iters']

        #Create folder system 
        if not os.path.exists(series_path):
            os.mkdir(series_path)
            for folders in ["samples","errors","distros","models","data"]:
                os.mkdir(os.path.join(series_path,folders))
        else:
            if not loading:
                pass 
            else:
                print(f"Loading")
                stash_dict          = json.loads(open(os.path.join(series_path,"data","data.txt"),"r").read())
                t.training_errors   = stash_dict['training_errors']
                t.training_classes  = stash_dict['training_classes']
                t.saved_state       = stash_dict


        t.override              = True

        t.lambda_gp             = lambda_gp
        t.lr_adjust             = True
        t.stop_thresh           = .9
        t.start_thresh          = .65
        t.stopped               = False

        #Check output sizes are kosher 
        inpv2                   = modelG.generate_rand(bs=1)
        print(f"MODELS: {name}\tn_z:{n_z}")
        if loading:
            root                    = os.path.join(series_path,"models")
            print(f"loading from epoch {root}")
            t.load_models(os.path.join(root,f"D_SAVE"),os.path.join(root,f"G_SAVE"))
            print("loaded models")
            t.epoch_num             = 0
        else:
            t.epoch_num             = 0
        print(f"G stats:\tout  - {   G.forward(inpv2).shape }\n        \tsize - {(sum([p.nelement()*p.element_size() for p in G.parameters()])/1000000):.2f}MB\n        \tparams: - {(sum([p.numel() for p in G.parameters()])/1000000):.2f}M")
        print(f"D stats:\tout  - {D( G.forward(inpv2)).shape}\n        \tsize - {(sum([p.nelement()*p.element_size() for p in D.parameters()])/1000000):.2f}MB\n        \tparams: - {(sum([p.numel() for p in D.parameters()])/1000000):.2f}M\n        \tval  - {D(G.forward(inpv2))[0].detach().cpu().item():.3f}")

        #Create optims 
        #optim_d = torch.optim.SGD(D.parameters(),lr=lrs[0],momentum=momentum)#lrs[0],betas=(momentum,.999))
        #optim_g = torch.optim.SGD(G.parameters(),lr=lrs[1],momentum=momentum)#lrs[1],betas=(momentum,.999))
    
        optim_g = torch.optim.AdamW(modelG.parameters(),lrs[1],betas=(momentum,.999))#,momentum=momentum)
        optim_d = torch.optim.AdamW(modelD.parameters(),lrs[0],betas=(momentum,.999))#,momentum=momentum)

        #optim_d                 = torch.optim.SGD(D.parameters(),lrs[0],weight_decay=wd,momentum=momentum,nesterov=True)
        #optim_g                 = torch.optim.SGD(G.parameters(),lrs[1],weight_decay=wd,momentum=momentum,nesterov=True)
        
        
        # t.outsize        = outsize
        # t.n_z            = n_z
        # t.set_learners(optim_d,optim_g,torch.nn.BCELoss())
        
        t.c_exec(load,t.max_iters,bs,optim_d,optim_g,n_z,outsize,series_path,50,verbose=True,sample_rate=1)
        print(f"done")

        #t.train_gen_on_real(filenames=files,bs=16,series_path=series_path)