# Outlier Dimensions Encode Task-Specific Knowledge
*by William Rudman, Catherine Chen, and Carsten Eickhoff*

Representations from large language models (LLMs) are known to be dominated by a small subset of dimensions with exceedingly high variance.
Previous works have argued that although ablating these \textit{outlier dimensions} in LLM representations hurts downstream performance, outlier dimensions are detrimental to the representational quality of embeddings. 
In this study, we investigate how fine-tuning impacts outlier dimensions and show that 
1) outlier dimensions that occur in pre-training persist in fine-tuned models and 
2) a single outlier dimension can complete downstream tasks with a minimal error rate.
 Our results suggest that outlier dimensions can encode crucial task-specific knowledge and that the value of a representation 
 in a single outlier dimension drives downstream model decisions.

This paper was accepted as a main conference paper at EMNLP 2023: 

## Using this Repo
### Step 1) Create Python environment and install dependencies: 

`pip install -r requirements.text`

### Step 2) Fine-tune models

We fine-tune all of our models using `run_model.py`. Below is an example of how to fine-tune BERT on SST-2. Full hyperparameters and random seeds are available in Section A of the Appendix. 

`python3 run_model.py --task sst2 --model bert --batch_size 32 --learning_rate 3e-5 --num_epochs 2 --seed 1`

Launching `run_model.py` will train and save the model to the PATH: 

`"models/" + self.config.model_name + "_" + str(self.config.seed) + "_" + self.config.task + ".pth"`. 

#### Argparser Options
+ **Task:** `sst2, qnli, qqp, mrpc, rte`.
+ **Model:** `bert, distbert, albert, roberta, gpt2, pythia-70m, pythia-160m, pythia-410m`.

### Step 3) Analysis 

There are a variety of different analyses that we run on our fine-tuned models. Similarly to run_model.py, `analysis.py` is controlled entirely by an argparser and uses the same `--task`, `--model`, and `--seed` options as run_model.py to load the correct model weights. The key difference is the `--analysis` flag that controls what tests we run on the fine-tuned models. Below is a description of all the tests we run in our paper. 

#### Analysis Tests
+ **brute_force_classification:** learns a linear threshold on the principal outlier dimension (EQN 1 in Section 3.2) and gives the 1-D accuracy of the principal outlier dimension.
+ **all_1d:**  applies our brute-force algorithm on every dimension. 
+ **save_finetuned_states:** save sentence embeddings on the validation data for a fine-tuned model. 
+ **save_pretrained_states:** saves sentence embedding on the validation data for a model that has not been fine-tuned for the given task. 
+ **avg_states:** averages the states across all random seeds the models were fine-tuned on.

### Step 4) Plotting the Results






