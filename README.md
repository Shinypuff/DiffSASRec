# DiffSASRec: Diffusion-based Sequential Recommendation

The following repository is the implementation of a diffusion SASRec model.

## Overview

The repository provides two main model variants:
- Original SASRec based on [pmixer's PyTorch implementation](https://github.com/pmixer/SASRec.pytorch) 
<img src="https://github.com/user-attachments/assets/7a1fc846-3d15-43a9-9789-09169f778af9" width=500>

- Diffusion-based Language Modeling inspired by [LLaDA](https://github.com/ML-GSAI/LLaDA):
  - Additional mask token embedding
  - Forward diffusion process to add noise to sequences
  - Reverse diffusion process for generative recommendation

### Training

To train the diffusion model, use the following command:

```bash
python main.py \
    --data_path your_data.csv \
    --train_dir experiment_name \
    --model_type diffusion \
    --num_recs 10 \
    --batch_size 128 \
    --maxlen 200 \
    --hidden_units 50 \
    --num_blocks 2 \
    --num_heads 2 \
    --dropout_rate 0.2 \
    --device cuda
```

Key parameters:
- `--data_path`: Path to your input data CSV file
- `--train_dir`: Directory to save model checkpoints and logs
- `--model_type`: Choose between 'vanilla' (original SASRec) or 'diffusion'
- `--diffusion_type`: Choose between 'multi' or 'single' for diffusion and topK inference respectively
- `--num_recs`: Number of recommendations (mask tokens for diffusion inference or K in topK)
- `--maxlen`: Maximum sequence length
- `--hidden_units`: Hidden dimension size
- `--num_blocks`: Number of transformer blocks
- `--num_heads`: Number of attention heads
- `--SFT`: Enable supervised fine-tuning after diffusion pretraining

### Data Format

The input data should be a CSV file with the following columns (default names can be customized through the argument parameters):
- UserId (`--users_col`)
- ProductId (`--items_col`)
- Timestamp (`--time_col`)

## Training Process

**Diffusion Pretraining**:
Similarly to LLaDA, our implementation defines a model distribution $p_{\theta}(x_0)$ through a *forward process* and a *reverse process*. With $t \in (0,1)$, the forward process generates partially masked sequence $x_t$, with each token from $x_0$ being masked with probability $t$ or remaining unmasked with probability $1 - t$. Thus, the distribution of masked tokens is:

$$
q_{t|0}(x_t^i|x_0^i) = 
    \begin{cases} 
    1 - t, & x_t^i = x_0^i, \\
    t, & x_t^i = \text{M (mask token)}.
    \end{cases}
$$

The predictor of DiffSASRec is a parametric model $p_{\theta}(\cdot|x_t)$ that takes $x_t$ as input and predicts all masked tokens simultaneously. It is trained using a cross-entropy loss computed only on the masked tokens:

$$
L(\theta) = -E_{t, x_0, x_t} \left[ \frac{1}{t} \sum_{i=1}^{L} 1[x_t^i = \mathbf{M}] \log p_{\theta}(x_0^i | x_t) \right]
$$

Thus, the training algorithm is the following:
![image](https://github.com/user-attachments/assets/2311d083-5e06-42c1-a4a8-0b3495b3347b)

## Inference

The inference is based on the *reverse process*: given a user interaction history $p_0$, we recover the data distribution by iteratively predicting masked tokens as t moves from 1 to 0. 

However, our objective is to provide K recommendations so that the next relevant item is present in our predictions. Thus, there are 2 ways to sample recommendations:

- **Single-step inference**: Predicts the next item directly. Top K logits are considered to compute metrics @K.

<img src="https://github.com/user-attachments/assets/cf489fd2-74fe-45c7-a414-50774d9ee6ed" width="500" />
    
- **Multi-step inference (diffusion-like)**: The algorithm progressively replaces K masked tokens in an iterative manner. At each step, it predicts possible values for the masked positions and assigns confidence scores to these predictions. Only the tokens with confidence scores exceeding a predefined threshold are updated in the sequence. If no predictions meet this threshold, the confidence requirement is gradually lowered.

<img src="https://github.com/user-attachments/assets/593f0a6d-6430-483d-b65a-32dd59527ebb" width="600" />

The multi-step inference procedure is presented in the Algorithm 2:

![image](https://github.com/user-attachments/assets/203bebd4-3079-4dcf-8488-0fb205b85e66)

## Data split

Repository provides a [time-based split](https://github.com/Shinypuff/DiffSASRec/blob/main/utils.py#L315) to simulate realistic sequential recommendation settings. The time-based splitting strategy involves defining a time cutoff (e.g. the 95th percentile mark) of the dataset.

To determine the holdout item, the first interaction of each user after the time split is considered. However, this item is only chosen if both the user and the item were present in the dataset before the split. If the first item does not meet this requirement—either because it is a new item that did not appear in the training set or because the users had no prior interactions with it, it is skipped, and the next interaction of the user is checked. This process continues until a suitable holdout item is found, ensuring that every user in the evaluation set has prior interactions and that the model has seen the selected item during training. The time-based splitting strategy is presented below:

<img src="https://github.com/user-attachments/assets/46ca1131-3c94-4168-8288-caffb323f3c7" width="600">

## Evaluation

The model is evaluated using standard recommendation metrics:
- NDCG@10
- HR@10
- MRR@10
- Coverage
