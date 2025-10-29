import os
import yaml
from yacs.config import CfgNode as CN

# General
_C = CN()
_C.model_name = 'gpt2_tiny'
_C.dataset_name = 'HuggingFaceFW/fineweb-edu'
_C.optimizer_name = 'Adam'
_C.wandb_project = 'mulo-testing'
_C.output_dir = './output'
_C.iters_max = 5000
_C.gas = 4
_C.batch_size = 32
_C.prefetch_factor = 32
_C.nbatch_log = 100
_C.num_workers = 8
_C.init_lr = 6e-4
_C.seed = 42
_C.local_rank = 0
_C.compile = True
_C.use_lr_scheduler = False
_C.warmup_ratio = 0.05
_C.suffix = ''
_C.use_mixed_precision = True
_C.MODEL = CN()


# GPT2
_C.MODEL.GPT2 = CN()

_C.MODEL.GPT2.attn_type = 'fused'
_C.MODEL.GPT2.model_type = 'gpt2'
_C.MODEL.GPT2.vocab_size = 50257
_C.MODEL.GPT2.block_size = 1024 #sequence length
_C.MODEL.GPT2.embd_pdrop = 0.1
_C.MODEL.GPT2.resid_pdrop = 0.1
_C.MODEL.GPT2.attn_pdrop = 0.1

_C.MODEL.MuGPT2 = CN()
_C.MODEL.GPT2.mulo_init = True
_C.MODEL.MuGPT2.encoder_var = 1.0
_C.MODEL.MuGPT2.output_mult = 1.0
_C.MODEL.MuGPT2.base_width_mult = 1.0
_C.MODEL.MuGPT2.width_mult = 1.0
_C.MODEL.MuGPT2.bsh_savepath = './mugpt2_base_shape.bsh'


#Optimizer
_C.OPTIM = CN()
_C.OPTIM.opt_checkpoint_path = None
_C.OPTIM.name = None
_C.OPTIM.max_grad_norm = None
_C.OPTIM.weight_decay = 0.0


#SGD
_C.OPTIM.SGD = CN()
_C.OPTIM.SGD.momentum = 0.9

# Adam
_C.OPTIM.Adam = CN()

# AdamW
_C.OPTIM.AdamW = CN()
_C.OPTIM.AdamW.weight_decay = 0.01
_C.OPTIM.AdamW.betas = (0.9,0.95)

# MuLO
_C.OPTIM.MuLO = CN()
_C.OPTIM.MuLO.step_mult = 0.01
_C.OPTIM.MuLO.cpkt_path = './MuLO_global_step5000_torch.pth'


# VeLO (legacy - use VeLO_naive or VeLO_CUDA instead)
_C.OPTIM.VeLO = CN()
_C.OPTIM.VeLO.num_steps = 5000 # will be adjusted automatically in train.py by num_steps=len(train_loader)*epochs
_C.OPTIM.VeLO.cpkt_path = './VeLO_torch.pth'

# VeLO_naive
_C.OPTIM.VeLO_naive = CN()
_C.OPTIM.VeLO_naive.num_steps = 5000 # will be adjusted automatically in train.py by num_steps=len(train_loader)*epochs
_C.OPTIM.VeLO_naive.cpkt_path = './VeLO_torch.pth'

# VeLO_CUDA
_C.OPTIM.VeLO_CUDA = CN()
_C.OPTIM.VeLO_CUDA.num_steps = 5000 # will be adjusted automatically in train.py by num_steps=len(train_loader)*epochs
_C.OPTIM.VeLO_CUDA.cpkt_path = './VeLO_CUDA_torch.pth'

def get_config():
    config = _C.clone()
    return config