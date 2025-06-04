
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \
train.py \
--config 'config' \
--override optimizer_name MuLO_cuda \
model_name gpt2_small \
init_lr 1.0 \
compile True \
gas 2 \
batch_size 32 \
suffix "_10B_tokens_FINAL_qkv_sched" \
iters_max 19073 \
use_lr_scheduler True \
use_mixed_precision True \
MODEL.GPT2.attn_type 'separate_kqv' \
OPTIM.max_grad_norm 1.0 \
seed $1

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \
train.py \
--config 'config' \
--override optimizer_name MuLO_cuda \
model_name gpt2_small \
init_lr 2.0 \
compile True \
gas 2 \
batch_size 32 \
suffix "_10B_tokens_FINAL_qkv_sched" \
iters_max 19073 \
use_lr_scheduler True \
use_mixed_precision True \
MODEL.GPT2.attn_type 'separate_kqv' \
OPTIM.max_grad_norm 1.0 \
seed $1


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \
train.py \
--config 'config' \
--override optimizer_name MuLO_cuda \
model_name gpt2_small \
init_lr 4.0 \
compile True \
gas 2 \
batch_size 32 \
suffix "_10B_tokens_FINAL_qkv_sched" \
iters_max 19073 \
use_lr_scheduler True \
use_mixed_precision True \
MODEL.GPT2.attn_type 'separate_kqv' \
OPTIM.max_grad_norm 1.0 \
seed $1


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \
train.py \
--config 'config' \
--override optimizer_name MuLO_cuda \
model_name gpt2_small \
init_lr 8.0 \
compile True \
gas 2 \
batch_size 32 \
suffix "_10B_tokens_FINAL_qkv_sched" \
iters_max 19073 \
use_lr_scheduler True \
use_mixed_precision True \
MODEL.GPT2.attn_type 'separate_kqv' \
OPTIM.max_grad_norm 1.0 \
seed $1


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \
train.py \
--config 'config' \
--override optimizer_name MuLO_cuda \
model_name gpt2_small \
init_lr 16.0 \
compile True \
gas 2 \
batch_size 32 \
suffix "_10B_tokens_FINAL_qkv_sched" \
iters_max 19073 \
use_lr_scheduler True \
use_mixed_precision True \
MODEL.GPT2.attn_type 'separate_kqv' \
OPTIM.max_grad_norm 1.0 \
seed $1


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




CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \
train.py \
--config 'config' \
--override optimizer_name MuLO_cuda \
model_name gpt2_small \
init_lr 0.75 \
compile True \
gas 2 \
batch_size 32 \
suffix "_10B_tokens_FINAL_qkv_sched" \
iters_max 19073 \
use_lr_scheduler True \
use_mixed_precision True \
MODEL.GPT2.attn_type 'separate_kqv' \
OPTIM.max_grad_norm 1.0 \
seed $1


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \
train.py \
--config 'config' \
--override optimizer_name MuLO_cuda \
model_name gpt2_small \
init_lr 0.5 \
compile True \
gas 2 \
batch_size 32 \
suffix "_10B_tokens_FINAL_qkv_sched" \
iters_max 19073 \
use_lr_scheduler True \
use_mixed_precision True \
MODEL.GPT2.attn_type 'separate_kqv' \
OPTIM.max_grad_norm 1.0 \
seed $1




CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \
train.py \
--config 'config' \
--override optimizer_name MuLO_cuda \
model_name gpt2_small \
init_lr 0.25 \
compile True \
gas 2 \
batch_size 32 \
suffix "_10B_tokens_FINAL_qkv_sched" \
iters_max 19073 \
use_lr_scheduler True \
use_mixed_precision True \
MODEL.GPT2.attn_type 'separate_kqv' \
OPTIM.max_grad_norm 1.0 \
seed $1



CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \
train.py \
--config 'config' \
--override optimizer_name MuLO_cuda \
model_name gpt2_small \
init_lr 0.1 \
compile True \
gas 2 \
batch_size 32 \
suffix "_10B_tokens_FINAL_qkv_sched" \
iters_max 19073 \
use_lr_scheduler True \
use_mixed_precision True \
MODEL.GPT2.attn_type 'separate_kqv' \
OPTIM.max_grad_norm 1.0 \
seed $1


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \
train.py \
--config 'config' \
--override optimizer_name MuLO_cuda \
model_name gpt2_small \
init_lr 0.01 \
compile True \
gas 2 \
batch_size 32 \
suffix "_10B_tokens_FINAL_qkv_sched" \
iters_max 19073 \
use_lr_scheduler True \
use_mixed_precision True \
MODEL.GPT2.attn_type 'separate_kqv' \
OPTIM.max_grad_norm 1.0 \
seed $1