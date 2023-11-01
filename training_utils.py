#PyTorch & Numpy
import torch 
from torch.utils.data import DataLoader 
import numpy as np
#import evaluate 

#HuggingFace Stuff 
from transformers import AutoTokenizer, AutoModel
from transformers import GPT2TokenizerFast, GPT2ForSequenceClassification
from transformers import BertTokenizerFast, BertForSequenceClassification,  BertForQuestionAnswering
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification,  DistilBertForQuestionAnswering
from transformers import AlbertTokenizerFast,  AlbertForSequenceClassification, AlbertForQuestionAnswering
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import GPTNeoXForSequenceClassification, GPTNeoXForQuestionAnswering
from transformers import AdamW
from datasets import load_dataset, load_metric  
from transformers import default_data_collator
import evaluate
import collections

# Pytorch stuff for DDP
from torch.utils.data.distributed import DistributedSampler 
from torch.nn.parallel import DistributedDataParallel as DDP 
import torch.multiprocessing as mp

# Misc  
import argparse 
import random 
import os
import wandb

# Custom Functions 
#from analysis import * 

# NOTE: Need to turn off Distributed sampling if you're only using a single GPU 
def prepare_dataloader(config, dataset, is_eval=False): 
    if is_eval:   
        dl = DataLoader(
                dataset,
                batch_size = config.batch_size, 
                pin_memory=True, 
                shuffle=False,
                collate_fn=classification_collate_fn, 
                num_workers=2)  
    else:  
        dl = DataLoader(
                dataset,
                batch_size = config.batch_size, 
                pin_memory=True, 
                sampler=DistributedSampler(dataset),
                shuffle=False,
                collate_fn=classification_collate_fn, 
                num_workers=2)    
    return dl 

def sow_seeds(seed):
    #Sow seeds 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) 
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    return None

########################### Classification Specific Functions ###########################
def classification_collate_fn(batch):
    """ Collate function used to make batches for the DataLoader"""  
    max_len = max([len(f["input_ids"]) for f in batch])  
    #Pad examples in the batch to be the same len 
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    # Retrieve labels 
    labels = [f["labels"] for f in batch]
    # Tensors need to be floats, labels need to long 
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.long)
    # Make the batch into a nice dictionary of input_ids, the attention mask and labels    
    outputs = { "input_ids": input_ids, "attention_mask": input_mask, "labels": labels }
    return outputs


def classification_eval(config, eval_loader, model):
    # Set model to eval mode. Load metric and create data loader.  
    model.eval() 
    num_saved_points = 0

    if config.task == "stsb": 
        pearsonr_metric = evaluate.load("pearsonr")
    # Send model to gpu if available
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    # Lists to store results. 
    preds_list = []
    labels_list = [] 
    
    for idx, batch in enumerate(eval_loader):
        # send batch to device  
        batch = {key: value.to(device) for key, value in batch.items()}  
        # Set model to eval and run input batches with no_grad to disable gradient calculations    
        with torch.no_grad():
            outputs = model(**batch, output_hidden_states=save_states) 
            logits = outputs.logits   
            
        # Store Predictions and Labels
        preds = logits.argmax(axis=1)        
        preds = preds.detach().cpu().numpy()  
        preds_list.append(preds)  
        labels = batch["labels"].detach().cpu().numpy() 
        states_list["labels"].append(labels) 
        labels_list.append(labels)  
    # Compute Accuracy 
    preds = np.concatenate(preds_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    # Set model to train!  
    model.train() 
    perf = (preds ==labels).sum()/len(preds)
    
    return perf

def load_classification_objs(config):
    """ 
    1) loads the specified dataset then preprocesses/tokenizes both the train and the eval data so it can easily be fed into a DataLoader. 
    2) load model specified in config.  
    3) load the optimizer. 
    """ 
    # Preprocessing and tokenizing data
    task_keys = {
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    }
     
    # Loading data for the given task  
    data = load_dataset("glue", config.task)
    num_labels = len(data["train"].features["label"].names)

    # Loading the specified model AND tokenizer 
    if "gpt2" in config.model_name: 
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2") 
        tokenizer.padding_side = "right"
        tokenizer.pad_token = tokenizer.eos_token                            
        model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=num_labels) 
        model.config.pad_token_id = model.config.eos_token_id
    
    if config.model_name == "bert": 
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased") 
        tokenizer.padding_side = "right"        
        model = BertForSequenceClassification.from_pretrained("bert-base-cased", num_labels=num_labels) 
         
    if config.model_name == "distbert":  
        tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-cased") 
        tokenizer.padding_side = "right"        
        model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-cased", num_labels=num_labels)
    
    if config.model_name == "albert":   
        tokenizer = AlbertTokenizerFast.from_pretrained("albert-base-v2") 
        tokenizer.padding_side = "right"        
        model = AlbertForSequenceClassification.from_pretrained("albert-base-v2", num_labels=num_labels) 

    if config.model_name == "roberta":   
        tokenizer =  RobertaTokenizer.from_pretrained("roberta-base")
        tokenizer.padding_side = "right"        
        model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=num_labels) 
    
    if "pythia" in config.model_name: 
        model = GPTNeoXForSequenceClassification.from_pretrained(
            "EleutherAI/" + config.model_name + "-deduped", num_labels=num_labels,
            )               
        tokenizer = AutoTokenizer.from_pretrained(
            "EleutherAI/" + config.model_name + "-deduped", 
            ) 
        # Setting  
        tokenizer.padding_side = "right"
        tokenizer.pad_token = tokenizer.eos_token 
        model.config.pad_token_id = model.config.eos_token_id
        if torch.cuda.is_available():
            model.cuda()
        
    # Preprocessing and tokenizing data
    def classification_preprocess(example): 
        key1, key2 = task_keys[config.task] 
        if key2 is None: inputs = (example[key1],)
        else: 
            inputs = (example[key1], example[key2])

        results = tokenizer(*inputs, max_length=256, truncation=True, add_special_tokens=True) 
        results["labels"] = example["label"] #if "label" in example else 0  
        return results    
     
    # For debugging purposes, set train equal to "mini" 
    if config.training == "Mini": 
        train_data = list(map(classification_preprocess, data["train"]))[:64]  
        eval_data = list(map(classification_preprocess, data["validation"]))[:32] 
    else: 
        train_data = list(map(classification_preprocess, data["train"]))  
        eval_data = list(map(classification_preprocess, data["validation"])) 

    optimizer =  torch.optim.AdamW(model.parameters(), lr=config.learning_rate) #load optimizer and learning rate. 
    return model,train_data, eval_data, optimizer

def classification_collate_fn(batch):
    """ Collate function used to make batches for the DataLoader"""  
    max_len = max([len(f["input_ids"]) for f in batch])  
    #Pad examples in the batch to be the same len 
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    # Retrieve labels 
    labels = [f["labels"] for f in batch]
    # Tensors need to be floats, labels need to long 
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.long)
    # Make the batch into a nice dictionary of input_ids, the attention mask and labels    
    outputs = { "input_ids": input_ids, "attention_mask": input_mask, "labels": labels }
    return outputs
