export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=10
export PYLO_DATA_DIR=$HOME/data


torchrun --nproc_per_node=4 train.py \
    --model vit_small_patch16_224 \
    --data-dir $PYLO_DATA_DIR/imagenette2 \
    --dataset imagenette \
    --experiment velo_vit_base_imagenette2 \
    --batch-size 32 \
    --epochs 100 \
    --workers 4 \
    --opt velo \
    --lr 1.0 \
    --sched none \
    --warmup-epochs 5 \
    --img-size 224 \
    --crop-pct 0.95 \
    --smoothing 0.0 \
    --clip-grad 1.0 \
    --mixup 0.1 \
    --cutmix 1.0 \
    --aa "rand-m7-mstd0.5" \
    --drop 0.1 \
    --drop-path 0.1 \
    --weight-decay 0.0 \
    --aug-repeats 4 \
    --hflip 0.5 \
    --seed 42 \
    --log-wandb \
    --wandb-project "pylo_examples" \
    --grad-accum-steps 1 \
    --amp \
    --torchcompile inductor