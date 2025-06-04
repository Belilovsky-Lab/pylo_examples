# module purge

# source /home/mila/b/benjamin.therien/pylo_install/setup.sh
# cd /home/mila/b/benjamin.therien/pylo_install/pylo-examples/gpt-training


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \
train.py \
--config 'config' \
--override optimizer_name AdamW \
model_name gpt2_small \
init_lr 0.001 \
compile True \
gas 2 \
batch_size 32 \
suffix "_10B_tokens_adamw_cosine_FINAL" \
iters_max 19073 \
use_lr_scheduler True \
warmup_ratio 0.02 \
use_mixed_precision True \
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
suffix "_10B_tokens_FINAL" \
iters_max 19073 \
use_lr_scheduler False \
use_mixed_precision True \
OPTIM.max_grad_norm 1.0 \
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
suffix "_10B_tokens_FINAL" \
iters_max 19073 \
use_lr_scheduler False \
use_mixed_precision True \
OPTIM.max_grad_norm 1.0 \
seed $1

