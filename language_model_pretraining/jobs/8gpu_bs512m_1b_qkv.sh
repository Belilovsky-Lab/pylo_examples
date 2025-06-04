

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \
train.py \
--config 'config' \
--override optimizer_name MuLO_cuda \
model_name gpt2_1b_w2048_d16_h32 \
init_lr 1.0 \
compile True \
gas 16 \
batch_size 4 \
suffix "_10B_tokens_FINAL_qkv" \
iters_max 19073 \
use_lr_scheduler False \
use_mixed_precision True \
OPTIM.max_grad_norm 1.0 \
MODEL.GPT2.attn_type 'separate_kqv' \
seed $1

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \
train.py \
--config 'config' \
--override optimizer_name VeLO \
model_name gpt2_1b_w2048_d16_h32 \
init_lr 1.0 \
compile True \
gas 16 \
batch_size 4 \
suffix "_10B_tokens_FINAL_qkv" \
iters_max 19073 \
use_lr_scheduler False \
use_mixed_precision True \
OPTIM.max_grad_norm 1.0 \
MODEL.GPT2.attn_type 'separate_kqv' \
seed $1

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \
train.py \
--config 'config' \
--override optimizer_name AdamW \
model_name gpt2_1b_w2048_d16_h32 \
init_lr 0.001 \
compile True \
gas 16 \
batch_size 4 \
suffix "_10B_tokens_adamw_cosine_FINAL_qkv" \
iters_max 19073 \
use_lr_scheduler True \
warmup_ratio 0.02 \
use_mixed_precision True \
MODEL.GPT2.attn_type 'separate_kqv' \
seed $1

