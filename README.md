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
Step 1) Create python environment and install dependencies: 
'pip install -r requirements.text'

