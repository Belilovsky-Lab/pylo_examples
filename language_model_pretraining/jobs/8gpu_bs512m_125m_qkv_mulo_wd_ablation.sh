


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \
train.py \
--config 'config' \
--override optimizer_name MuLO_cuda \
model_name gpt2_small \
init_lr 1.0 \
compile True \
gas 2 \
batch_size 32 \
suffix "_10B_tokens_FINAL_qkv_sched_wd=0.0001" \
iters_max 19073 \
use_lr_scheduler False \
use_mixed_precision True \
MODEL.GPT2.attn_type 'separate_kqv' \
OPTIM.max_grad_norm 1.0 \
OPTIM.weight_decay 0.0001 \
seed 42



CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \
train.py \
--config 'config' \
--override optimizer_name MuLO_cuda \
model_name gpt2_small \
init_lr 1.0 \
compile True \
gas 2 \
batch_size 32 \
suffix "_10B_tokens_FINAL_qkv_sched_wd=0.001" \
iters_max 19073 \
use_lr_scheduler False \
use_mixed_precision True \
MODEL.GPT2.attn_type 'separate_kqv' \
OPTIM.max_grad_norm 1.0 \
OPTIM.weight_decay 0.001 \
seed 42



CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \
train.py \
--config 'config' \
--override optimizer_name MuLO_cuda \
model_name gpt2_small \
init_lr 1.0 \
compile True \
gas 2 \
batch_size 32 \
suffix "_10B_tokens_FINAL_qkv_sched_wd=0.01" \
iters_max 19073 \
use_lr_scheduler False \
use_mixed_precision True \
MODEL.GPT2.attn_type 'separate_kqv' \
OPTIM.max_grad_norm 1.0 \
OPTIM.weight_decay 0.01 \
seed 42



CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \
train.py \
--config 'config' \
--override optimizer_name MuLO_cuda \
model_name gpt2_small \
init_lr 1.0 \
compile True \
gas 2 \
batch_size 32 \
suffix "_10B_tokens_FINAL_qkv_sched_wd=0.1" \
iters_max 19073 \
use_lr_scheduler False \
use_mixed_precision True \
MODEL.GPT2.attn_type 'separate_kqv' \
OPTIM.max_grad_norm 1.0 \
OPTIM.weight_decay 0.1 \
seed 42



CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \
train.py \
--config 'config' \
--override optimizer_name MuLO_cuda \
model_name gpt2_small \
init_lr 1.0 \
compile True \
gas 2 \
batch_size 32 \
suffix "_10B_tokens_FINAL_qkv_sched_wd=0.00001" \
iters_max 19073 \
use_lr_scheduler False \
use_mixed_precision True \
MODEL.GPT2.attn_type 'separate_kqv' \
OPTIM.max_grad_norm 1.0 \
OPTIM.weight_decay 0.00001 \
seed 42




CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \
train.py \
--config 'config' \
--override optimizer_name MuLO_cuda \
model_name gpt2_small \
init_lr 1.0 \
compile True \
gas 2 \
batch_size 32 \
suffix "_10B_tokens_FINAL_qkv_sched_wd=0.000001" \
iters_max 19073 \
use_lr_scheduler False \
use_mixed_precision True \
MODEL.GPT2.attn_type 'separate_kqv' \
OPTIM.max_grad_norm 1.0 \
OPTIM.weight_decay 0.000001 \
seed 42