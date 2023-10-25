# Outlier Dimensions Encode Task-Specific Knowledge
*by William Rudman, Catherine Chen, and Carsten Eickhoff*

Representations from large language models (LLMs) are known to be dominated by a small subset of dimensions with exceedingly high variance.
Previous works have argued that although ablating these \textit{outlier dimensions} in LLM representations hurts downstream performance, outlier dimensions are detrimental to the representational quality of embeddings. 
In this study, we investigate how fine-tuning impacts outlier dimensions and show that 
1) outlier dimensions that occur in pre-training persist in fine-tuned models and 
2) a single outlier dimension can complete downstream tasks with a minimal error rate.
 Our results suggest that outlier dimensions can encode crucial task-specific knowledge and that the value of a representation 
 in a single outlier dimension drives downstream model decisions. 

## Using this Repo
### Step 1) Create Python environment and install dependencies: 

`pip install -r requirements.text`

### Step 2) Fine-tune models

We fine-tune all of our models using `run_model.py`. Launching `run_model.py` will train and save the model to the PATH: `"models/" + self.config.model_name + "_" + str(self.config.seed) + "_" + self.config.task + ".pth"`. Below is an example of how to fine-tune BERT on SST-2. Full hyperparameters and random seeds are available in Section A of the Appendix. 

`python3 run_model.py --task sst2 --model bert --batch_size 32 --learning_rate 3e-5 --num_epochs 2 --seed 1`

#### Argparser Options
+ **Task:** `sst2, qnli, qqp, mrpc, rte`
+ **Model:** `bert, distbert, albert, roberta, gpt2, pythia-70m, pythia-160m, pythia-410m`

### Step 3) Analysis 





