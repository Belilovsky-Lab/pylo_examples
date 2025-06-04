# module purge

# source /home/mila/b/benjamin.therien/pylo_install/setup.sh
# cd /home/mila/b/benjamin.therien/pylo_install/pylo-examples/gpt-training





# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \
# train.py \
# --config 'config' \
# --override optimizer_name VeLO \
# model_name gpt2_small \
# init_lr 1.0 \
# compile True \
# gas 2 \
# batch_size 8 \
# suffix "_10B_tokens_FINAL_qkv_wd_0.1" \
# iters_max 19073 \
# use_lr_scheduler False \
# use_mixed_precision True \
# OPTIM.max_grad_norm 1.0 \
# OPTIM.weight_decay 0.1 \
# MODEL.GPT2.attn_type 'separate_kqv' \
# seed $1



# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \
# train.py \
# --config 'config' \
# --override optimizer_name VeLO \
# model_name gpt2_small \
# init_lr 1.0 \
# compile True \
# gas 2 \
# batch_size 8 \
# suffix "_10B_tokens_FINAL_qkv_wd_0.01" \
# iters_max 19073 \
# use_lr_scheduler False \
# use_mixed_precision True \
# OPTIM.max_grad_norm 1.0 \
# OPTIM.weight_decay 0.01 \
# MODEL.GPT2.attn_type 'separate_kqv' \
# seed $1




# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \
# train.py \
# --config 'config' \
# --override optimizer_name VeLO \
# model_name gpt2_small \
# init_lr 1.0 \
# compile True \
# gas 2 \
# batch_size 8 \
# suffix "_10B_tokens_FINAL_qkv_wd_0.001" \
# iters_max 19073 \
# use_lr_scheduler False \
# use_mixed_precision True \
# OPTIM.max_grad_norm 1.0 \
# OPTIM.weight_decay 0.001 \
# MODEL.GPT2.attn_type 'separate_kqv' \
# seed $1


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \
# train.py \
# --config 'config' \
# --override optimizer_name VeLO \
# model_name gpt2_small \
# init_lr 1.0 \
# compile True \
# gas 2 \
# batch_size 8 \
# suffix "_10B_tokens_FINAL_qkv_wd_0.0001" \
# iters_max 19073 \
# use_lr_scheduler False \
# use_mixed_precision True \
# OPTIM.max_grad_norm 1.0 \
# OPTIM.weight_decay 0.0001 \
# MODEL.GPT2.attn_type 'separate_kqv' \
# seed $1




CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \
train.py \
--config 'config' \
--override optimizer_name VeLO \
model_name gpt2_small \
init_lr 1.0 \
compile True \
gas 2 \
batch_size 8 \
suffix "_10B_tokens_FINAL_qkv_wd_0.00001" \
iters_max 19073 \
use_lr_scheduler False \
use_mixed_precision True \
OPTIM.max_grad_norm 1.0 \
OPTIM.weight_decay 0.00001 \
MODEL.GPT2.attn_type 'separate_kqv' \
seed $1

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \
train.py \
--config 'config' \
--override optimizer_name VeLO \
model_name gpt2_small \
init_lr 1.0 \
compile True \
gas 2 \
batch_size 8 \
suffix "_10B_tokens_FINAL_qkv_wd_0.000001" \
iters_max 19073 \
use_lr_scheduler False \
use_mixed_precision True \
OPTIM.max_grad_norm 1.0 \
OPTIM.weight_decay 0.000001 \
MODEL.GPT2.attn_type 'separate_kqv' \
seed $1