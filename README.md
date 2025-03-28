# DiffSASRec: Diffusion-based Sequential Recommendation

The following repository is the implementation of a diffusion SASRec model.

## Overview

The model builds on two key components:
- Original SASRec implementation
- Diffusion-based Language Modeling inspired by LLaDA

### Training

To train the diffusion model, use the following command:

```bash
python new_diffusion_main.py \
    --data_path your_data.csv \
    --train_dir experiment_name \
    --model_type diffusion \
    --num_masks 10 \
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
- `--num_masks`: Number of mask tokens for diffusion inference (the @K value)
- `--maxlen`: Maximum sequence length
- `--hidden_units`: Hidden dimension size
- `--num_blocks`: Number of transformer blocks
- `--num_heads`: Number of attention heads
- `--SFT`: Enable supervised fine-tuning after diffusion pretraining

### Data Format

The input data should be a CSV file with the following columns (default names can be customized through the argument parameters):
- UserId
- ProductId
- Timestamp

## Training Process

1. **Diffusion Pretraining**:
   - Random masking of sequence tokens
   - Training to predict masked tokens
   - Adaptive masking rates

2. **(Optional) Supervised Fine-tuning**:
   - Cross-entropy loss on target items
   - L2 regularization on embeddings
   - Adam optimizer with custom learning rate

## Evaluation

The model is evaluated using standard recommendation metrics:
- NDCG@10
- HR@10
- MRR@10
- Coverage
