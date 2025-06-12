# Welcome to PyLO Examples!


## PyLO Evals (This Repository)

PyLO Evals complements [PyLO](https://github.com/Belilovsky-Lab/pylo) by providing benchmark code for standard pre-training tasks that are of interest to the ML community. Key features include:

- Implementations of practical, full-scale training runs (not just toy problems)
- Benchmarks comparing learned optimizers against state-of-the-art optimizers like AdamW with cosine annealing
- Reproducible evaluation protocols for fair comparison
- Examples showing how to integrate PyLO optimizers into real training workflows

While learned optimization has traditionally been tested on small, artificial tasks that don't represent practical training scenarios, PyLO-Examples enables rigorous evaluation on realistic workloads that matter to practitioners.



# Installation 

Run the following code:
```
install_dir=$PWD/torch_lo_install
mkdir $install_dir
cd $install_dir

wget https://repo.anaconda.com/miniconda/Miniconda3-py311_24.7.1-0-Linux-x86_64.sh
bash Miniconda3-py311_24.7.1-0-Linux-x86_64.sh -b -p $PWD/miniconda3
source $PWD/miniconda3/bin/activate

# Install custom MUP for MuLO support
pip install git+https://github.com/bentherien/mup.git

# Install pylo with CUDA support
git clone https://github.com/belilovskylab/pylo.git
cd pylo
pip install .
python setup.py install --cuda 

# For logging Set WANDB environment variables
export WANDB_API_KEY=YOUR_KEY
export WANDB_PROJECT=pylo_examples
export WANDB_MODE=online

```

# GPT Training

Here are some example commands for running single-GPU meta-training.


## Training a GPT model with MuLO [https://arxiv.org/abs/2406.00153](https://arxiv.org/abs/2406.00153)
```
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 \
train.py --config 'config' \
--override \
optimizer_name MuLO_cuda \
compile True \
init_lr 2
```

## Training a GPT model with VeLO [https://arxiv.org/abs/2406.00153](https://arxiv.org/abs/2211.09760)
```
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 \
train.py --config 'config' \
--override \
optimizer_name veLO \
compile True \
init_lr 2
```

## Training a GPT model with AdamW and cosine annealing [https://arxiv.org/abs/2211.09760](https://arxiv.org/abs/1711.05101)
```
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 \
train.py --config 'config' \
--override \
optimizer_name AdamW \
compile True \
init_lr 2
```

# ViT Training

## Training ViT with MuLO [https://arxiv.org/abs/2406.00153](https://arxiv.org/abs/2406.00153)
```
CUDA_VISIBLE_DEVICES=0 
```

## Training ViT with VeLO [https://arxiv.org/abs/2406.00153](https://arxiv.org/abs/2211.09760)
```
CUDA_VISIBLE_DEVICES=0 torchrun 
```

## Training ViT with AdamW and cosine annealing [https://arxiv.org/abs/2211.09760](https://arxiv.org/abs/1711.05101)
```
CUDA_VISIBLE_DEVICES=0 torchrun 
```


# Citation

If you use PyLO in your research, please consider citing our work:

```bibtex
@software{pylo2025,
  author = {Paul Janson, Benjamin Therien, Quentin Anthony, Xialong Huang, Abhinav Moudgil and Eugene Belilovsky},
  title = {PyLo: A PyTorch Library for Learned Optimizers},
  year = {2025},
  url = {https://github.com/Belilovsky-Lab/pylo}
}
```
