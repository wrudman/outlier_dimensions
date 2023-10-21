
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from training_utils import * 

import argparse
import torch 
from torch.utils.data import DataLoader 
import numpy as np
# Pytorch stuff for DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP 
import torch.multiprocessing as mp
# Misc  
from tqdm import tqdm 
import argparse 
from datasets import Dataset
# Custom Functions 
from analysis import * 
from training_utils import *
import hickle


def id_estimate(id_model, data):
    # Note: this function only works for the built-in skdim.id estimators
    return id_model().fit(data).dimension_

def collate_fn(batch):
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

def flatten_list(nested_list):
    flattened_list = []
    for sublist in nested_list:
        if isinstance(sublist, list):
            flattened_list.extend(flatten_list(sublist))
        else:
            flattened_list.append(sublist)
    return flattened_list

def run_analysis(config, model, eval_loader, seed):  
    # Used to compute stats of the model during the course of training  
    #id_model = TwoNN  
    #performance, _, sentence_states, preds = classification_eval(config, eval_data, model, save_states=True, sentence_embed=True)  

    eval_states, eval_labels = get_states(config, model, eval_loader) 

    states = {"states": eval_states, "labels": eval_labels}
    hickle.dump(states, config.model_name + "_" +  str(seed) + "_" + config.task + "_states.hickle", mode='w')  
    
    results = {} 
    # Store performance  
    results["mean"] = [] 
    results["std"] = [] 
    # Stack all states for a sigle vector space of all representations  
    points=np.vstack(eval_states)
    points=torch.tensor(points) 
    
    results["mean"].append(torch.mean(points, axis=0)) 
    results["std"].append(torch.std(points, axis=0))  

    hickle.dump(results, config.model_name + "_" +  str(seed) + "_" + config.task + "_analysis.hickle", mode='w')  
    return results

# Eval ablate 
def classification_eval_ablate(args, data, model, epoch, ablate_dims, direction, hidden_states=False):
    # Set model to eval mode. Load metric and create data loader.  
    print("Evaluating") 
    model.eval() 
    eval_loader = DataLoader(data, batch_size=args.batch_size, collate_fn=collate_fn)
    
    # Send model to gpu if available
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    # Lists to store results. 
    preds_list = []
    labels_list = [] 
    states_list = {"states": [], "labels": [], "preds": []}  
    # main EVAL loop 
    
    for idx, batch in enumerate(eval_loader):
        # send batch to device  
        batch = {key: value.to(device) for key, value in batch.items()} 
       
        # Set model to eval and run input batches with no_grad to disable gradient calculations   
        model.eval()
        with torch.no_grad():
            outputs = model(**batch, output_hidden_states=True) 
            states = outputs.hidden_states  
            if args.model == "gpt2": 
                states = states[-1][:,-1,:]
            elif args.model == "bert": 
                states = states[-1][:,0,:]
            # Set the ablating dimensions to zero
            # Let's try to do this by multiplying.           
            for dim in ablate_dims:
                states[:,dim] = 0  
            # Feed in the ablating representations into the classification head to get logits.  
            if args.model == "gpt2": 
                logits = model.score(states)
            elif args.model == "bert":  
                #In the BERT model, there is an additional pooling layer before the classifier  
                pool = model.bert.pooler(states.unsqueeze(dim=1)) 
                logits = model.classifier(pool) 
            # CAN COLLECT HIDDEN STATES HERE. 
            if hidden_states==True:
                states_list["states"].append(states.detach().cpu().numpy()) 

        # Store Predictions and Labels 
        preds = logits.argmax(axis=1)        
        preds = preds.detach().cpu().numpy()  
        preds_list.append(preds)   
        states_list["preds"].append(preds) 
        labels = batch["labels"].detach().cpu().numpy()  
        states_list["labels"].append(labels)   
        labels_list.append(labels)  
        probs = torch.nn.functional.softmax(logits, dim=1)  
    
    # Compute Accuracy 
    preds = np.concatenate(preds_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    acc = (preds == labels).sum()/len(preds)
    # NEED TO CHANGE THE NAME OF THE FILES SO WE CAN HAVE TOP DOWN AND BOTTOM DOWN. 
    if hidden_states==True: 
        hickle.dump(states_list, args.model + "_" + args.task + "_" + epoch + "_" + str(len(ablate_dims))+ "_" + direction +".hickle", mode='w')
    
    return acc, probs 

# Brute threshold and logistic regression analysis
def get_states(config, model, data_loader):
    # Set model to eval mode. Load metric and create data loader.  
    model.eval() 
    num_saved_points = 0
    # Send model to gpu if available
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    # Lists to store results. 
    states =  []
    labels = []
    
    for _, batch in enumerate(data_loader):
        # send batch to device  
        batch = {key: value.to(device) for key, value in batch.items()}  
        # Set model to eval and run input batches with no_grad to disable gradient calculations   
        with torch.no_grad():
            outputs = model(**batch, output_hidden_states=True) 
            batch_states = outputs.hidden_states 
            # Get sentence embeddings
            # CLS token for encoders
            if config.model_name in ["bert", "distbert", "albert", "roberta"]:     
                states.append(torch.reshape(batch_states[-1][:,0,:], (-1,768)).detach().cpu().numpy()) 
            # Last layer representation for decoders
            if config.model_name in ["gpt2", "gpt2-medium", "pythia-70m", "pythia-160m", "pythia-410m"]:        
                states.append(torch.reshape(batch_states[-1][:,-1,:],(-1, batch_states[-1].shape[-1])).detach().cpu().numpy())    
        # Also, append the labels             
        labels.append(batch["labels"].detach().cpu().numpy()) 
        #num_saved_points += np.vstack(states).shape[0] 
        #if num_saved_points > 75000:
        #break
    # Return states and labels (to make life easier)
    return np.vstack(states), np.concatenate(labels)

def max_var_predict(config, model, train_loader, eval_loader): 
    # GOAL: get the perfomance on a downstream task based on the max outlier dimension 
    # LEARNING OPTIMAL THRESHOLD
    results = {} 
    # collect sentence representations from training data.    
    train_states, train_labels = get_states(config, model, train_loader)

    # find the outlier dimensions with the maximum variance  
    var = np.var(train_states, axis=0) 
    max_dim = np.argmax(var)
    # HERE I CAN ARGSORT THE VARIANCE, ITERATE THROUGH THEN GO THROUGH BF ALG
    # Let's write a new function... there I will go through each dimension and assess the 1d perfomance
    # Then, we can
    # a) graph the association beetwen magnitude and 1d perf ability  
    # b) create a plot with acc and dim index sorted by the magnitude of variance. 
    # STORING VAR STATS  
    results["max_var"] = var[max_dim]
    results["mean_var"] = np.mean(var) 
    results["max_dim"] = max_dim 
    
    # Representing sentence embeddings only by max_dim
    d1_train_states = train_states[:,max_dim]
    
    # Use logistic regression to learn a threshold
    linear_model = LogisticRegression(max_iter=500)
    linear_model.fit(np.reshape(d1_train_states, (-1,1)), train_labels)
    
    # Brute force approach to finding optimal threshold.
    # We start at the mean, and then move in intervals of +/- 0.5 between mean the of d1_states - 5 and between mean of the d1_states + 5. 
    threshold = np.mean(d1_train_states)
    best_threshold = 0
    best_acc = 0
    # rule is 0 if the relationship 0 for less than threshold and 1 for greater than threshold
    rule = 0
    
    for i in np.linspace(-25, 25, 101): 
    #for i in np.linspace(-75,75,901):
        # write a rule where we predict class 0 if below or equal to threshold and class 1 above threshold. 
        preds = np.where(d1_train_states <= threshold + i, 0, 1)
        acc = accuracy_score(preds, train_labels)
        if max(acc, 1-acc) > best_acc: 
            best_acc = max(acc, 1-acc)
            best_threshold = threshold + i
            rule = np.argmax([acc, 1-acc])
    
    # Storing results 
    results["best_threshold"] = best_threshold
    results["rule"] = rule 
    
    # EVALUATING
    # collect values at max dim along with labels for the eval data
    eval_states, eval_labels = get_states(config, model, eval_loader)
    # Using same max_dim learned from train data! 
    d1_eval_states = eval_states[:,max_dim]
     
    #Logistic Regression evaluation
    lg_preds = linear_model.predict(np.reshape(d1_eval_states, (-1,1)))
    if config.task == "cola":
        #matthews_metric = evaluate.load("matthews_correlation")
        r = matthews_metric.compute(references=eval_labels, predictions=lg_preds)
        lg_perf = r["matthews_correlation"]
    else:
        lg_perf = accuracy_score(lg_preds, eval_labels) 

    # Evaluate using optimal brute force threshold.
    if rule == 0:
        bf_preds = np.where(d1_eval_states <= best_threshold, 0, 1)
    if rule == 1:
        bf_preds = np.where(d1_eval_states > best_threshold, 0, 1) 
    
    if config.task == "cola":
        matthews_metric = evaluate.load("matthews_correlation")
        r = matthews_metric.compute(references=eval_labels, predictions=bf_preds)
        bf_perf = r["matthews_correlation"]
    else:
        bf_perf = accuracy_score(bf_preds, eval_labels)
    
    results["lg_perf"] = lg_perf
    results["bf_perf"] = bf_perf
    return results

# Eval ablate 
def classification_eval_ablate(config, eval_loader, model, ablate_dims):
    # Set model to eval mode. Load metric and create data loader.
    model.eval() 

    # Send model to gpu if available
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    # Lists to store results. 
    preds_list = []
    labels_list = [] 
    # main EVAL loop 
    
    for _, batch in enumerate(eval_loader):
        # send batch to device  
        batch = {key: value.to(device) for key, value in batch.items()} 
       
        # Set model to eval and run input batches with no_grad to disable gradient calculations   
        model.eval()
        with torch.no_grad():
            outputs = model(**batch, output_hidden_states=True) 
            states = outputs.hidden_states  
            if config.model_name == "gpt2": 
                states = states[-1][:,-1,:]
            elif config.model_name in ["bert", "albert", "distbert"]: 
                states = states[-1][:,0,:]
            # Set the ablating dimensions to zero
            # Let's try to do this by multiplying.           
            for dim in ablate_dims:
                states[:,dim] = 0  
            # Feed in the ablating representations into the classification head to get logits.  
            if config.model_name == "gpt2": 
                logits = model.score(states)
            elif config.model_name == "bert":  
                #In the BERT model, there is an additional pooling layer before the classifier  
                pool = model.bert.pooler(states.unsqueeze(dim=1)) 
                logits = model.classifier(pool) 
            elif config.model_name == "albert":
                pool= model.albert.pooler(states)
                pool_activations = model.albert.pooler_activation(pool)
                logits = model.classifier(pool_activations)
            elif config.model_name == "distbert":
                pre_classifier = model.pre_classifier(states)
                logits = model.classifier(pre_classifier)

        # Store Predictions and Labels 
        preds = logits.argmax(axis=1)        
        preds = preds.detach().cpu().numpy()  
        preds_list.append(preds)   
        labels = batch["labels"].detach().cpu().numpy()   
        labels_list.append(labels)  
        probs = torch.nn.functional.softmax(logits, dim=1)  
    
    # Compute Accuracy 
    preds = np.concatenate(preds_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    acc = (preds == labels).sum()/len(preds)    
    return acc #,probs

def all_var_predict(config, model, train_loader, eval_loader):
    # Dictionary to store variance, perfomrance and the dimension index.  
    results = {}
    results["var"] = []
    results["perf"] = []
    results["dim"] = [] 
    thresholds = []
    rules = []

    # Get sentence embeddings of the training data along with its class label! 
    train_states, train_labels = get_states(config, model, train_loader) 

    # Calculate variance of sentence embeddings
    var = np.var(train_states, axis=0) 

    # Go through dims in terms of variance.
    sort_idx = np.argsort(var) 
    for idx in sort_idx: 
        # Get the order of dims in terms of variance 
        results["dim"].append(idx)
        results["var"].append(var[idx])

        # Representing sentence embeddings only by max_dim
        d1_train_states = train_states[:,idx]
        
        # Brute force approach to finding optimal threshold.
        # We start at the mean, and then move in intervals of +/- 0.5 between mean the of d1_states - 25 and between mean of the d1_states + 25. 
        threshold = np.mean(d1_train_states)
        best_threshold = 0
        best_acc = 0
        # rule is 0 if the relationship 0 for less than threshold and 1 for greater than threshold
        rule = 0
        
        #for i in np.linspace(-25,25,101): 
        for i in np.linspace(-75,75,901):
            # write a rule where we predict class 0 if below or equal to threshold and class 1 above threshold. 
            preds = np.where(d1_train_states <= threshold + i, 0, 1)
            acc = accuracy_score(preds, train_labels)
            if max(acc, 1-acc) > best_acc: 
                best_acc = max(acc, 1-acc)
                best_threshold = threshold + i
                rule = np.argmax([acc, 1-acc])
        thresholds.append(best_threshold) 
        rules.append(rule)
        
    
    # EVALUATING BF THRESHOLD
    #Get sentence embeddings on the validation data along with the class label. 
    eval_states, eval_labels = get_states(config, model, eval_loader)
    
    for i, idx in enumerate(sort_idx): 
        # Get 1d eval states.
        d1_eval_states = eval_states[:,idx]
        
        # Evaluate using optimal brute force threshold.
        if rules[i] == 0:
            bf_preds = np.where(d1_eval_states <= thresholds[i], 0, 1)
        if rules[i] == 1:
            bf_preds = np.where(d1_eval_states > thresholds[i], 0, 1) 
        results["perf"].append(accuracy_score(bf_preds, eval_labels))  
    return results 



def main():
    # Argparser to create config 
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default=1, type=str)
    parser.add_argument("--training", default="True", type=str) 
    parser.add_argument("--task", default="sst2", type=str)
    parser.add_argument("--num_epochs", default=1, type=int)
    parser.add_argument("--batch_size", default=32, type=int) 
    parser.add_argument("--seed", default=1, type=int) 
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--direction", default="top", type=str)
    parser.add_argument("--analysis", default="max_var", type=str)
    config = parser.parse_args()   
     
    if config.analysis == "brute_force_classification":
        model_perf = [] 
        lg_perf = [] 
        bf_perf = [] 
        threshold = [] 
        rule = [] 
        # CHANGE BACK TO 1,2,3,4 
        for i in [1,2,3,4]: 
            # Let's test out max_var_predict!!! 
            model, train_data, eval_data, _ = load_classification_objs(config) 
            # load model weights and prep dataloaders
            model.load_state_dict(torch.load("models/" +  config.model_name + "_" + str(i) + "_" + config.task + ".pth"))   
            train_loader = prepare_dataloader(config, train_data, is_eval=True) 
            eval_loader = prepare_dataloader(config, eval_data, is_eval=True)   
            # full model eval acc
            perf = classification_eval(config, eval_loader, model, save_states=False, sentence_embed=False)
            model_perf.append(perf) 
            # Run max_var_predict
            results = max_var_predict(config, model, train_loader, eval_loader)
            lg_perf.append(results["lg_perf"])
            bf_perf.append(results["bf_perf"])
            threshold.append(results["best_threshold"])
            rule.append(results["rule"]) 
        print("------------------------------------") 
        print("------------------------------------")  
        print("RESULTS FOR: ", config.model_name) 
        print(config.task) 
        print("------------------------------------") 
        print("------------------------------------")  
        print("ACC", model_perf) 
        print("MODEL MEAN", np.mean(model_perf)) 
        print("MODEL STD", np.std(model_perf)) 
        print("------------------------------------") 
        print("LG", lg_perf) 
        print("LG MEAN", np.mean(lg_perf))
        print("LG STD", np.std(lg_perf)) 
        print("------------------------------------") 
        print("BF", bf_perf) 
        print("BF MEAN", np.mean(bf_perf)) 
        print("BF STD", np.std(bf_perf)) 
        print("BEST THRESHOLDS:", threshold) 
        print("RULES:", rule) 
    if config.analysis == "eval_ablate": 
        for i in [1,2,3,4]: 
            model, train_data, eval_data, _ = load_classification_objs(config) 
            # load model weights and prep dataloaders
            model.load_state_dict(torch.load("models/" +  config.model_name + "_" + str(i) + "_" + config.task + ".pth")) 
            train_loader = prepare_dataloader(config, train_data, is_eval=True) 
            eval_loader = prepare_dataloader(config, eval_data, is_eval=True)   
            
            train_states, _ = get_states(config, model, train_loader)
            var = np.diagonal(np.cov(train_states.T)) 
            outlier_order = np.argsort(var) 
            
            if config.direction=="top":
                outlier_order = outlier_order[::-1]
            
            scores = []  
            for j in range(len(outlier_order)):
                ablate_dims = outlier_order[:j+1]  
                scores.append(classification_eval_ablate(config, eval_loader, model, ablate_dims))  
            print(scores) 
            np.savetxt(config.model_name + "_" + str(i) + "_" + config.task + "_" + config.direction + ".txt", scores, delimiter=',')
    
    if config.analysis == "run_analysis":
        for i in [1,2,3,4]: 
            model, _, eval_data, _ = load_classification_objs(config) 
            # load model weights and prep dataloaders
            model.load_state_dict(torch.load("models/" +  config.model_name + "_" + str(i) + "_" + config.task + ".pth")) 
            eval_loader = prepare_dataloader(config, eval_data, is_eval=True) 
            run_analysis(config, model, eval_loader, seed=i)  
            print("analysis run for seed: ", str(i))
    
    if config.analysis == "save_finetuned_states":  
        for i in [1,2,3,4]: 
            model, _, eval_data, _ = load_classification_objs(config) 
            # load model weights and prep dataloaders
            model.load_state_dict(torch.load("models/" +  config.model_name + "_" + str(i) + "_" + config.task + ".pth")) 
            eval_loader = prepare_dataloader(config, eval_data, is_eval=True)
            states, _ = get_states(config, model, eval_loader)   
            hickle.dump(states, config.model_name + "_" + str(i) + "_" + config.task + "_states.hickle", mode='w') 
    
    if config.analysis == "save_pretrained_states":  
        model, _, eval_data, _ = load_classification_objs(config) 
        # load model weights and prep dataloaders 
        eval_loader = prepare_dataloader(config, eval_data, is_eval=True)
        states, _ = get_states(config, model, eval_loader)   
        hickle.dump(states, config.model_name + "_" + config.task + "_pretrained_states.hickle", mode='w')
    
    if config.analysis == "avg_states":
        states = [] 
        for i in [1,2,3,4]: 
            f = hickle.load(config.model_name + "_" + str(i) + "_" + config.task + "_states.hickle")
            #states.append(f["states"])  
            states.append(f) 
        print("LEN STATES", len(states))
        print(states) 
        states = np.vstack(states) 
        print("SHAPE:", states.shape) 
        if config.model_name == "pythia-410m":
            states = np.reshape(states, (4,-1,1024))
        elif config.model_name == "pythia-70m":
            states = np.reshape(states, (4,-1,512)) 
        else:
            states = np.reshape(states, (4,-1,768))
        avg = np.mean(states, axis=0) 
        print(config.task, avg.shape)
        hickle.dump(avg, config.model_name + "_" + config.task + "_avg_states.hickle", mode='w') 
    
    if config.analysis == "all_1d":
        for i in [1,2,3,4]: 
            print("Running all 1D analysis") 
            # Let's test out max_var_predict!!! 
            model, train_data, eval_data, _ = load_classification_objs(config) 
            # load model weights and prep dataloaders
            model.load_state_dict(torch.load("models/" +  config.model_name + "_" + str(i) + "_" + config.task + ".pth")) 
            train_loader = prepare_dataloader(config, train_data, is_eval=True) 
            eval_loader = prepare_dataloader(config, eval_data, is_eval=True)   
            results = all_var_predict(config, model, train_loader, eval_loader) 
            print(results) 
            hickle.dump(results, config.model_name + "_" + str(i) + "_" + config.task + "_all_1d.hickle", mode='w')
            
if __name__ == '__main__':
    main()
