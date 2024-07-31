#PyTorch & Numpy
import torch 
from torch.utils.data import DataLoader 
import numpy as np
# Pytorch stuff for DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP 
import torch.multiprocessing as mp
# Misc  
import argparse 
import wandb
import typing
from datasets import Dataset
# Custom Functions 
from analysis import * 
from training_utils import *
import hickle

def ddp_setup(rank, world_size):
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

class Trainer:
    def __init__(
        self,
        config: dict,  
        model: torch.nn.Module, 
        train_data: DataLoader, 
        eval_data: DataLoader, 
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        raw_data: typing.Optional[Dataset] = None, 
        squad_val: typing.Optional[Dataset] = None 
        ) -> None:  
        self.config = config  
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.eval_data = eval_data 
        self.optimizer = optimizer
        self.best_performance = 0
        self.steps = 0 
        self.model = DDP(model, device_ids=[gpu_id]) 
        self.scaler = torch.cuda.amp.GradScaler()
        self.raw_data = raw_data
        self.squad_val = squad_val
     
    def _run_batch(self, batch):  
        # THIS IS WHERE I NEED TO CALL MY MONITOR TRAINING         
        with torch.cuda.amp.autocast(dtype=torch.float16): 
        #with torch.cuda.amp.autocast(dtype=torch.float32):
            outputs = self.model(**batch, output_hidden_states=False)    
            loss = outputs.loss 
        #NOTE lets train all of our models first except for 1 random seed then for the last random seed run training analysis 
        # Backprop (scaler for fp16 training) 
        self.scaler.scale(loss).backward()  
        self.scaler.step(self.optimizer) 
        self.scaler.update() 
        self.model.zero_grad()
        self.optimizer.zero_grad() 

        self.steps += 1

    def _run_epoch(self, epoch):

        """ 
        INPUT: the current epoch for a given gpu-id
        Sends mini-batches to device and calls _run_batch to complete a single epoch.
        """ 
        b_sz = len(next(iter(self.train_data))['input_ids']) 
        self.train_data.sampler.set_epoch(epoch)
        # Send everything to device and call run_batch 
        for _, batch in enumerate(self.train_data):
            batch = {key: value.to(self.gpu_id) for key, value in batch.items()}    
            self._run_batch(batch)
   
    def _save_model(self):
        """ 
        Saves model checkpoint to PATH 
        """ 
        PATH = "models/" + self.config.model_name + "_" + str(self.config.seed) + "_" + self.config.task + ".pth"
        torch.save(self.model.module.state_dict(), PATH)
        print("MODEL SAVED")
      
    def train(self): 
        """ 
        Train the model for num_epochs and save the model for the last epoch.  
        """ 
        #wandb.watch 
        for epoch in range(self.config.num_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0: 
                # WE PRINT THIS JUST TO MAKE SURE TRAINING IS HAPPENING. 
                acc = classification_eval(self.config, self.eval_data, self.model)   
                print("ACC", acc) 
                #wandb.log({"Accuracy": acc}) 
                        
                if epoch+1 == self.config.num_epochs: 
                    # SAVE MODEL AND RUN ANALYSIS 
                    print("DONE") 
                    self._save_model()               

def main(rank: int, config: dict, world_size: int):
    # Wandb init
    print(config) 
    # Monitor everything with wandb. NOTE: only logging metrics for GPU0. So, look at the results files and NOT these. This is just for monitoring experiments.   
    results = {}   
    #wandb.init(project=config.model_name + "_" + config.task + "_seeds", name=str(config.seed))  
    # Training with DDP
    ddp_setup(rank,world_size)   
    # Sow seeds  
    sow_seeds(int(config.seed))
    print("SEED", config.seed) 
    if config.task == 'squad': 
        model, train_data, eval_data, raw_data, optimizer = load_squad_objs(config) 
    else:
        model, train_data, eval_data, optimizer = load_classification_objs(config)
        
    # Create dataloaders 
    train_loader = prepare_dataloader(config, train_data) 
    eval_loader = prepare_dataloader(config, eval_data, is_eval=True)  

    trainer = Trainer(config, model, train_loader, eval_loader, optimizer, rank)
    
    trainer.train()
    destroy_process_group()

if __name__  == "__main__":
    # Argparser to create config 
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", default=2, type=int)
    parser.add_argument("--batch_size", default=32, type=int) 
    parser.add_argument("--task", default="sst2", type=str)
    parser.add_argument("--model_name", default="bert", type=str)
    parser.add_argument("--seed", default=1, type=int)
    # --training also takes the argument "Mini" which will train the model on a very small subset for debugging purposes.
    parser.add_argument("--training", default="True", type=str) 
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    config = parser.parse_args() 
    
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(config, world_size), nprocs=world_size) 


