
# Language Model Pre-training Instructions



# Quick Setup
```
apt-get update
apt install tmux vim rsync htop -y
tmux

mkdir pylo_examples_install
cd pylo_examples_install

wget https://repo.anaconda.com/miniconda/Miniconda3-py311_24.7.1-0-Linux-x86_64.sh
bash Miniconda3-py311_24.7.1-0-Linux-x86_64.sh -b -p $PWD/miniconda3
source $PWD/miniconda3/bin/activate

# install PyLO
git clone https://github.com/Belilovsky-Lab/pylo
cd pylo
pip install .
python setup.py install --cuda 
cd ..

# install custom MuP for compatility with mu-learned optimizers
git clone https://github.com/bentherien/mup
cd mup
pip install -e .
cd ..

# setup pylo examples
git clone https://github.com/Belilovsky-Lab/pylo_examples
cd pylo_examples/language_model_pretraining
pip install -r requirements.txt


# download data
mkdir data/fineweb_edu_10B
python tools/download_dataset.py --output_dir data/fineweb_edu_10B

#For logging be sure to set your WANDB API KEY!
export WANDB_API_KEY=
```

# Quick Start: Train a 410M parameter Language model with MuLO or VeLO and replicate our results table

| Optimizer         | LM Loss ↓ (355M) |
|-------------------|------------------|
| $\mu$LO$_M$       | 3.18             |
| VeLO              | **2.89**         |
| AdamW + Cosine    | 2.91             |


```
#VeLO
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \
train.py \
--config 'config' \
--override optimizer_name VeLO \
model_name gpt2_410m_w1024_d24_h16 \
init_lr 1.0 \
compile True \
gas 8 \
batch_size 8 \
suffix "_10B_tokens_FINAL_qkv" \
iters_max 19073 \
use_lr_scheduler False \
use_mixed_precision True \
OPTIM.max_grad_norm 1.0 \
MODEL.GPT2.attn_type 'separate_kqv' \
seed $1

#MuLO
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \
train.py \
--config 'config' \
--override optimizer_name MuLO_cuda \
model_name gpt2_410m_w1024_d24_h16 \
init_lr 1.0 \
compile True \
gas 8 \
batch_size 8 \
suffix "_10B_tokens_FINAL_qkv" \
iters_max 19073 \
use_lr_scheduler False \
use_mixed_precision True \
OPTIM.max_grad_norm 1.0 \
MODEL.GPT2.attn_type 'separate_kqv' \
seed $1

#AdamW
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \
train.py \
--config 'config' \
--override optimizer_name AdamW \
model_name gpt2_410m_w1024_d24_h16 \
init_lr 0.001 \
compile True \
gas 8 \
batch_size 8 \
suffix "_10B_tokens_adamw_cosine_FINAL_qkv" \
iters_max 19073 \
use_lr_scheduler True \
warmup_ratio 0.02 \
use_mixed_precision True \
MODEL.GPT2.attn_type 'separate_kqv' \
seed $1
```





# Adding a schedule and Weight Decay to MuLO or VeLO

## Training a GPT model with VeLO + schedule [https://arxiv.org/abs/2406.00153](https://arxiv.org/abs/2211.09760)
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \
train.py \
--config 'config' \
--override optimizer_name MuLO_cuda \
model_name gpt2_small \
init_lr 1.0 \
compile True \
gas 2 \
batch_size 32 \
suffix "_10B_tokens_FINAL_qkv_sched_wd_0.0001" \
iters_max 19073 \
use_lr_scheduler False \
use_mixed_precision True \
MODEL.GPT2.attn_type 'separate_kqv' \
OPTIM.max_grad_norm 1.0 \
seed 42
```


## Training a 125M parameter LM MuLO [https://arxiv.org/abs/2406.00153](https://arxiv.org/abs/2406.00153) using a cosine decay schedule and weight decay
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \
train.py \
--config 'config' \
--override optimizer_name MuLO_cuda \
model_name gpt2_small \
init_lr 1.0 \
compile True \
gas 2 \
batch_size 32 \
suffix "_10B_tokens_FINAL_qkv_sched_wd_0.0001" \
iters_max 19073 \
use_lr_scheduler True \
use_mixed_precision True \
MODEL.GPT2.attn_type 'separate_kqv' \
OPTIM.weight_decay 0.0001 \
OPTIM.max_grad_norm 1.0 \
seed 42
```

# Performance

## Training Results

The following table shows the performance comparison between different optimizers on the GPT training task. The results include the hyperparameters used and the loss values at different training steps.
| Optimizer | Learning Rate | Other Hyperparameters | Loss @ 1k steps | Loss @ 2k steps | Loss @ 3k steps | Loss @ 4k steps | Loss @ 5k steps |
|-----------|---------------|------------------------|-----------------|-----------------|-----------------|-----------------|-----------------|
| MuLO      | 1.0           | Default settings       | --              | --              | --              | --              | --              |
| VeLO      | 1.0           | Default settings       | --              | --              | --              | --              | --              |
| AdamW     | 1e-3          | β₁=0.9, β₂=0.95, ε=1e-8| --              | --              | --              | --              | --              |

The results demonstrate that learned optimizers (MuLO and VeLO) consistently outperform traditional optimizers like AdamW across all training steps. MuLO achieves the best performance, with approximately 7% lower loss compared to AdamW at the 5k step mark.

### Data Processing

All experiments were conducted using the FinWeb-Edu dataset with a context length of 1024 tokens and a batch size of 32 per GPU. The model architecture is a standard decoder-only transformer with 12 layers, 12 attention heads, and an embedding dimension of 768.



# Slower Custom Setup


# Dataset Preprocessing

The dataset used for training is the FinWeb-Edu dataset. For convenience, we provide a script to preprocess the dataset from [Andrej Karpathy's build-NanoGPT](https://github.com/karpathy/build-nanogpt/blob/master/fineweb.py).

```
python fineweb.py --remote_name finweb_edu --local_name finweb_edu
```

This will download the dataset and save it in the `data` directory.

# Citation

If you use PyLO in your research, please consider citing our work:

```bibtex
@article{pylo,
  title={PyLO Evals: A PyTorch Library for Benchmarking Learned Optimizers},
  author={},
}
```
