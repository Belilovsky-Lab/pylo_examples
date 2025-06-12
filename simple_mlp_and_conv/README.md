# Training MLPs and ResNets with PyLO Optimizers

This example demonstrates training Multi-Layer Perceptrons (MLPs) and ResNets using learned optimizers (VeLO and MuLO) and compares them against traditional optimizers like AdamW on smaller datasets.

## Install Environment

```bash
pip install -r requirements.txt
```

## Download Datasets

### CIFAR-10 and CIFAR-100
These datasets will be downloaded automatically when running training scripts.

### ImageNet-32x32
```bash
mkdir -p data
export PYLO_DATA_DIR=$PWD/data
pip install huggingface_hub
python tools/download_dataset.py --output_dir $PYLO_DATA_DIR --repo_id btherien/imagenet-32x32x3
```

## Quick Start: Train 128-width MLP on CIFAR-10

### VeLO [https://arxiv.org/abs/2211.09760](https://arxiv.org/abs/2211.09760)
```bash
python train.py \
    --opt velo \
    --model mlp \
    --dataset cifar10 \
    --data_dir $PYLO_DATA_DIR \
    --lr 1.0 \
    --epochs 100 \
    --batch-size 128 \
    --seed 42
```

### MuLO [https://arxiv.org/abs/2406.00153](https://arxiv.org/abs/2406.00153)
```bash
python train.py \
    --opt mulo \
    --model mlp \
    --dataset cifar10 \
    --data_dir $PYLO_DATA_DIR \
    --lr 1.0 \
    --epochs 100 \
    --batch-size 128 \
    --seed 42
```

### AdamW + Cosine Annealing (Baseline)
```bash
python train.py \
    --opt adamw \
    --model mlp \
    --dataset cifar10 \
    --data_dir $PYLO_DATA_DIR \
    --lr 1e-3 \
    --epochs 100 \
    --batch-size 128 \
    --seed 42
```

## Quick Start: Train 128-width MLP on ImageNet-32x32

### VeLO [https://arxiv.org/abs/2211.09760](https://arxiv.org/abs/2211.09760)
```bash
python train.py \
    --opt velo \
    --model mlp \
    --dataset imagenet32x32 \
    --data_dir $PYLO_DATA_DIR \
    --lr 1.0 \
    --epochs 100 \
    --batch-size 128 \
    --seed 42
```

### MuLO [https://arxiv.org/abs/2406.00153](https://arxiv.org/abs/2406.00153)
```bash
python train.py \
    --opt mulo \
    --model mlp \
    --dataset imagenet32x32 \
    --data_dir $PYLO_DATA_DIR \
    --lr 1.0 \
    --epochs 100 \
    --batch-size 128 \
    --seed 42
```

### AdamW + Cosine Annealing (Baseline)
```bash
python train.py \
    --opt adamw \
    --model mlp \
    --dataset imagenet32x32 \
    --data_dir $PYLO_DATA_DIR \
    --lr 1e-3 \
    --epochs 100 \
    --batch-size 128 \
    --seed 42
```

## Quick Start: Train ResNet-18 on CIFAR-10

### VeLO [https://arxiv.org/abs/2211.09760](https://arxiv.org/abs/2211.09760)
```bash
python train.py \
    --opt velo \
    --model resnet18 \
    --dataset cifar10 \
    --data_dir $PYLO_DATA_DIR \
    --lr 1.0 \
    --epochs 100 \
    --batch-size 128 \
    --seed 42
```

### MuLO [https://arxiv.org/abs/2406.00153](https://arxiv.org/abs/2406.00153)
```bash
python train.py \
    --opt mulo \
    --model resnet18 \
    --dataset cifar10 \
    --data_dir $PYLO_DATA_DIR \
    --lr 1.0 \
    --epochs 100 \
    --batch-size 128 \
    --seed 42
```

### AdamW + Cosine Annealing (Baseline)
```bash
python train.py \
    --opt adamw \
    --model resnet18 \
    --dataset cifar10 \
    --data_dir $PYLO_DATA_DIR \
    --lr 1e-3 \
    --epochs 100 \
    --batch-size 128 \
    --seed 42
```

## Quick Start: Train ResNet-50 on CIFAR-100

### VeLO [https://arxiv.org/abs/2211.09760](https://arxiv.org/abs/2211.09760)
```bash
python train.py \
    --opt velo \
    --model resnet50 \
    --dataset cifar100 \
    --data_dir $PYLO_DATA_DIR \
    --lr 1.0 \
    --epochs 100 \
    --batch-size 128 \
    --seed 42
```

### MuLO [https://arxiv.org/abs/2406.00153](https://arxiv.org/abs/2406.00153)
```bash
python train.py \
    --opt mulo \
    --model resnet50 \
    --dataset cifar100 \
    --data_dir $PYLO_DATA_DIR \
    --lr 1.0 \
    --epochs 100 \
    --batch-size 128 \
    --seed 42
```

### AdamW + Cosine Annealing (Baseline)
```bash
python train.py \
    --opt adamw \
    --model resnet50 \
    --dataset cifar100 \
    --data_dir $PYLO_DATA_DIR \
    --lr 1e-3 \
    --epochs 100 \
    --batch-size 128 \
    --seed 42
```

## Supported Datasets
- **cifar10**: CIFAR-10 dataset (10 classes, 32x32 images)
- **cifar100**: CIFAR-100 dataset (100 classes, 32x32 images)
- **imagenet32x32**: ImageNet dataset downsampled to 32x32 pixels

## Supported Models
- **mlp**: Multi-Layer Perceptron with 128 hidden units
- **resnet18**: ResNet-18 architecture
- **resnet50**: ResNet-50 architecture

## Supported Optimizers
- **velo**: VeLO (Vectorized Learned Optimizer)
- **mulo**: MuLO (μ-Parameterized Learned Optimizer)
- **adamw**: AdamW with optional cosine annealing
- **sgd**: SGD with momentum