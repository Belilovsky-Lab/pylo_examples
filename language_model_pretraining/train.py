import argparse
import importlib  # Unused
import os
import random
import time
import wandb
import tiktoken
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from mup import get_shapes, make_base_shapes, set_base_shapes
from pylo.optim import MuLO_naive, MuLO_CUDA
from pylo.optim.Velo_naive import VeLO_naive
from pylo.optim.velo_cuda import VeLO_CUDA
from tqdm import tqdm, trange
import wandb
from collections import defaultdict
import mugpt, gpt
from contextlib import nullcontext
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

def get_model(config):
    if 'Mu' in config.optimizer_name:
        model = getattr(mugpt, config.model_name)(config, wm=config.MODEL.MuGPT2.width_mult)
        base_model = getattr(mugpt, config.model_name)(config, wm=config.MODEL.MuGPT2.base_width_mult, type='base')
        delta_model = getattr(mugpt, config.model_name)(config, wm=config.MODEL.MuGPT2.base_width_mult+1, type='delta')
        # set the shapes to have base_width = 1 
        set_base_shapes(model, base_model, delta=delta_model,  override_base_dim=1)
        model.reset_parameters_all()
    else:
        model = getattr(gpt, config.model_name)(config)

    
    return model
def get_lr_scheduler(optimizer, config):
    """Create a learning rate scheduler with optional warmup and cosine decay using PyTorch implementations."""
    if not config.use_lr_scheduler:
        return None
    
    warmup_steps = int(config.iters_max * config.warmup_ratio)
    
    # Create warmup scheduler (linear warmup from 0.1 to init_lr)
    # start_factor must be > 0 and <= 1
    warmup_scheduler = LinearLR(
        optimizer, 
        start_factor=1e-8,  # Changed from 0.0 to 0.1 to avoid ValueError
        end_factor=1.0, 
        total_iters=warmup_steps
    )
    
    # Create cosine decay scheduler
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.iters_max - warmup_steps,
        eta_min=config.init_lr * 0.1
    )
    
    # Combine the two schedulers
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps]
    )
    
    return scheduler

def create_optim(config, model):
    if config.optimizer_name =='Adam':
        optimizer = optim.Adam(model.parameters(), lr=config.init_lr)
    elif config.optimizer_name=='AdamW':
        optimizer = optim.AdamW(model.parameters(), 
                                lr=config.init_lr, 
                                weight_decay=config.OPTIM.AdamW.weight_decay, 
                                betas=config.OPTIM.AdamW.betas)
    elif config.optimizer_name=='SGD':
        optimizer = optim.SGD(model.parameters(), lr=config.init_lr, momentum=config.OPTIM.SGD.momentum)
    elif config.optimizer_name=='MuLO_naive':
        optimizer = MuLO_naive(model.parameters(), 
                               lr=config.init_lr, 
                               max_grad_norm=config.OPTIM.max_grad_norm,
                                weight_decay=config.OPTIM.weight_decay)
    elif config.optimizer_name=='MuLO_cuda':
        optimizer = MuLO_CUDA(model.parameters(), 
                              lr=config.init_lr, 
                              max_grad_norm=config.OPTIM.max_grad_norm,
                              weight_decay=config.OPTIM.weight_decay,
                              load_from_file=config.OPTIM.opt_checkpoint_path)
    elif config.optimizer_name=='VeLO_naive':
        optimizer = VeLO_naive(model.parameters(),
                        lr=config.init_lr,
                        num_steps=config.iters_max,
                        weight_decay=config.OPTIM.weight_decay)
    elif config.optimizer_name=='VeLO_CUDA':
        optimizer = VeLO_CUDA(model.parameters(),
                        lr=config.init_lr,
                        num_steps=config.iters_max,
                        weight_decay=config.OPTIM.weight_decay,
                        legacy=False)
    elif config.optimizer_name=='VeLO':
        # Legacy option, redirects to VeLO_naive
        print("Warning: 'VeLO' is deprecated, using 'VeLO_naive' instead. Please use 'VeLO_naive' or 'VeLO_CUDA' explicitly.")
        optimizer = VeLO_naive(model.parameters(),
                        lr=config.init_lr,
                        num_steps=config.iters_max,
                        weight_decay=config.OPTIM.weight_decay)
    else:
        raise NotImplementedError(f"Unkown optimizer: {optimizer_name}")
    
    scheduler = get_lr_scheduler(optimizer, config)
    return optimizer, scheduler


def set_seed(seed=1, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = deterministic


class Timing:
    # Static dictionaries to store run times and historical stats
    run_times_dict = defaultdict(list)  # Stores the elapsed times for each named timer
    historical_stats = defaultdict(lambda: {"mean": [], "std": []})  # Stores the historical mean and std for each named timer

    def __init__(self,name):
        self.name = name

    def __enter__(self):
        self.start = time.time()
        return self  # This allows us to use "as x" in the with statement

    def __exit__(self, exc_type, exc_value, traceback):
        self.end = time.time()
        duration = self.end - self.start
        Timing.run_times_dict[self.name].append(duration)



def parse_args():
    def config_from_name(module_name):
        name = 'get_config'
        module = importlib.import_module(module_name)
        config = getattr(module, name)()
        return config
        
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description="Override YACS config from CLI")
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument(
       "-o", "--override",
        help="Modify config options using the command-line, e.g. MODEL.LR 0.01 TRAIN.BATCH_SIZE 64",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args, _ = parser.parse_known_args()
    config = config_from_name(args.config)
    
    # Merge command-line arguments into config
    if args.override:
        for k, v in zip(args.override[::2], args.override[1::2]):
            if v is not None:
                print0(f"[INFO] Overriding config value: {k}={v}")
        config.merge_from_list(args.override)
    
            
    return args, config

def cross_entropy(output, labels, _fp16=False):
    """From pretrain_gpt2:forward_step()"""
    """
    if self.fp16_lm_cross_entropy:
        assert output.dtype == torch.half
        loss = mpu.vocab_parallel_cross_entropy(output, labels)
    else:
        loss = mpu.vocab_parallel_cross_entropy(output.float(), labels)
        return loss
    """
    labels, loss_mask = labels[0], labels[1]
    if _fp16:
        assert output.dtype == torch.half and loss_mask.dtype == torch.half
        losses = mpu.vocab_parallel_cross_entropy(output.contiguous(), labels)
    else:
        losses = mpu.vocab_parallel_cross_entropy(output.float().contiguous(), labels)
    loss_mask = loss_mask.view(-1)
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
    return loss

def train(model, train_loader, optimizer, scheduler, config, run):
    criterion = nn.CrossEntropyLoss(ignore_index=-1).cuda()
    model.train()

    # Pre-fetch first batch
    next_idxs, next_labels = next(train_loader)
    next_idxs, next_labels = next_idxs.to(config.device, non_blocking=True), next_labels.to(config.device, non_blocking=True)

    pbar = tqdm(
        range(config.iters_max),
        initial=0,
        total=config.iters_max,
        ascii=True,
        desc="Training Steps",
    )
    for i in range(config.iters_max):
        # Using set_to_none=True is more memory-efficient and slightly faster
        # as it avoids memory allocation for zeros and the operation of writing zeros
        train_loss = torch.tensor(0.0, device=config.device)
        
        with Timing('fwbw T'):
            
            # gradient accumulation
            for gas_step in range(config.gas):
                # Use pre-fetched batch
                idxs, labels = next_idxs, next_labels
                # Overlap data transfer with computation
                with Timing('data T'):
                    next_idxs, next_labels = next(train_loader)
                    next_idxs, next_labels = next_idxs.to(config.device, non_blocking=True), next_labels.to(config.device, non_blocking=True)

                # Use no_sync for all but the last accumulation step to avoid unnecessary communication
                context = model.no_sync if gas_step < config.gas - 1 else nullcontext
                with context():
                    with Timing('fwd T'):
                        # Use bfloat16 for better training stability with mixed precision
                        with torch.cuda.amp.autocast(enabled=config.use_mixed_precision, dtype=torch.bfloat16):
                            preds = model(idxs)
                            loss = criterion(preds, labels.view(-1))
                    
                    with Timing('bwd T'):
                        # Use a persistent scaler for mixed precision
                        if config.use_mixed_precision:
                            # Create scaler once and reuse
                            if not hasattr(train, 'scaler'):
                                train.scaler = torch.cuda.amp.GradScaler()
                            scaled_loss = train.scaler.scale(loss)
                            scaled_loss.backward()
                        else:
                            loss.backward()

                    train_loss += loss.clone().detach() / config.gas


        # get loss
        with Timing('Loss AR'):
            dist.all_reduce(train_loss, op=dist.ReduceOp.AVG)
            mean_loss = train_loss
            torch.cuda.synchronize()
        
        #optimizer step
        with Timing('opt T'):
            # if config.optimizer_name in ['MuLO_naive', 'MuLO_cuda']:
            
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if config.optimizer_name in ['VeLO_naive', 'VeLO_CUDA', 'VeLO']:
                optimizer.step(mean_loss)
            else:
                optimizer.step()
        
        optimizer.zero_grad(set_to_none=True)
        
        # Update learning rate if scheduler is used
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = config.init_lr

        if config.rank == 0:
            mu = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            postfix = {
                # "iteration": i,
                "Train loss": mean_loss.item(),
                "gpu mem": mu,
                "shard": train_loader.current_shard,
                "lr": current_lr
            }
            postfix.update({k:v[-1] for k,v in Timing.run_times_dict.items()})
            pbar.set_postfix(postfix)
            run.log(postfix)
            pbar.update(1)

    return mean_loss


def load_tokens(filename):
    npt = np.load(filename, allow_pickle=True)
    npt = npt.astype(np.int32) # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # get the shard filenames
        data_root = "data/fineweb_edu_10B"

        # data_root = "/home/mila/b/benjamin.therien/github/modded-nanogpt/data/finewebedu10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        # print(shards)
        # exit(0)
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if process_rank == 0:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def __iter__(self):
        return self

    def __next__(self):
        return self.next_batch()

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            # print(f"loading next shard: {self.current_shard}")
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y
    


if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    rank = int(os.environ["RANK"])
    def print0(s):
        if rank == 0:
            print(s)
    args, config = parse_args()
    config.defrost()
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config.rank = rank
    config.world_size = int(os.environ['WORLD_SIZE'])
    config.OPTIM.VeLO.num_steps = config.iters_max
    config.OPTIM.VeLO_naive.num_steps = config.iters_max
    config.OPTIM.VeLO_CUDA.num_steps = config.iters_max
    # Set default values for scheduler if not present
    if not hasattr(config, 'use_lr_scheduler'):
        config.use_lr_scheduler = False
    if not hasattr(config, 'warmup_ratio'):
        config.warmup_ratio = 0.05  # Default 5% warmup
    config.freeze()
    print(rank, config.world_size)
    if config.rank == 0:
        print(config.dump())
    set_seed(config.seed)
    torch.cuda.set_device(config.rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    os.makedirs(config.output_dir, exist_ok=True)

    train_loader = DataLoaderLite(B=config.batch_size, 
                                  T=config.MODEL.GPT2.block_size, 
                                  process_rank=config.rank, 
                                  num_processes=config.world_size, 
                                  split="train")

    val_loader = DataLoaderLite(B=config.batch_size, 
                                T=config.MODEL.GPT2.block_size, 
                                process_rank=config.rank, 
                                num_processes=config.world_size, 
                                split="val")


    model = get_model(config)
    model.cuda()
    model = nn.parallel.DistributedDataParallel(model, device_ids=None, output_device=None)

    if config.compile:
        model = torch.compile(model)

    optimizer, scheduler = create_optim(config, model)

    if config.rank == 0:
        run = wandb.init(
            project=config.wandb_project,
            group=f"{config.dataset_name.replace('/', '_')}_{config.model_name}_{config.optimizer_name}_LR_{config.init_lr}_max-norm_{config.OPTIM.max_grad_norm}{config.suffix}",
            config=config,
        )
    else:
        run = None

    start_time = time.time()
    final_loss = train(model, train_loader, optimizer, scheduler, config, run)
    print(f"Final loss: {final_loss}")

    print0("="*100)
    print0("Training Complete")
    print0("="*100)
    dist.destroy_process_group()
