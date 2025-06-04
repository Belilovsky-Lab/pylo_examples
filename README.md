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


pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1
pip install tensorflow==2.19.0
pip install transformers==4.49.0 datasets==3.3.2 tokenizers==0.21.0
pip install numpy==2.0.2 pandas==2.2.3 matplotlib==3.9.4 seaborn==0.13.2
pip install timm==1.0.15 wandb==0.19.8
pip install tqdm==4.66.4 tiktoken==0.9.0
pip install yacs

# Install custom MUP for MuLO support
pip install git+https://github.com/bentherien/mup.git

git clone https://github.com/belilovskylab/pylo.git
pip install . --config-settings="--build-option=--cuda"
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

# ImageNet Training

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
