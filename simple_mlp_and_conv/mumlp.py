"""This is a simple MLP example using PyTorch.

The script is meant to replicate experiments from https://arxiv.org/abs/2406.00153, which were origninally run in Jax.
"""
# Standard library imports
import argparse
import math
import os
import time
import io
from pickletools import optimize
import multiprocessing

# Third-party imports
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm
import h5py
from PIL import Image

# MuP imports
from mup import MuSGD, MuReadout, get_shapes, make_base_shapes, set_base_shapes
import torch
import random
import numpy as np
import os
import time

from helpers import set_torch_seed, MupVarianceScaling, get_dataset
from pylo.optim import MuLO_naive,MuLO_CUDA,VeLO

class MuMLP(nn.Module):
    def __init__(
        self,
        in_channels,
        width=128,
        num_classes=10,
        nonlin=F.relu,
        bias=True,
        output_mult=1.0,
        input_mult=1.0,
        init_std=1.0,
        log_activations=True,
        use_mup=True,
    ):
        super(MuMLP, self).__init__()
        self.nonlin = nonlin
        self.input_mult = input_mult
        self.output_mult = output_mult
        self.init_std = init_std
        self.log_activations = log_activations
        self.fc_1 = nn.Linear(in_channels, width, bias=bias)
        self.fc_2 = nn.Linear(width, width, bias=bias)
        self.fc_3 = nn.Linear(width, width, bias=bias)
        # self.fc_4 = nn.Linear(width, num_classes, bias=bias)
        if use_mup:
            self.fc_4 = MuReadout(width, num_classes, bias=bias, output_mult=output_mult)
        else:
            self.fc_4 = nn.Linear(width, num_classes, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):

        ini = MupVarianceScaling(1.0, "fan_in", "truncated_normal")
        ini.initialize(self.fc_1.weight)
        ini.initialize(self.fc_2.weight)
        ini.initialize(self.fc_3.weight)
        nn.init.zeros_(self.fc_4.weight)
        nn.init.normal_(self.fc_1.bias, mean=0.0, std=1.0)
        nn.init.normal_(self.fc_2.bias, mean=0.0, std=1.0)
        nn.init.normal_(self.fc_3.bias, mean=0.0, std=1.0)
        nn.init.normal_(self.fc_4.bias, mean=0.0, std=1.0)

    # def forward(self, x):
    #     out = self.nonlin(self.fc_1(x.flatten(1)) * self.input_mult)
    #     out = self.nonlin(self.fc_2(out))
    #     out = self.nonlin(self.fc_3(out))
    #     return self.fc_4(out) 

    def forward(self, x):
        activations = {}  # Dictionary to store activation logs

        out = self.fc_1(x.flatten(1)) * self.input_mult
        if self.log_activations:
            with torch.no_grad():
                activations["layer_0_pre-act_l1"] = torch.mean(torch.abs(out)).item()
                activations["layer_0_pre-act"] = out.clone().detach()
        
        out = self.nonlin(out)
        if self.log_activations:
            with torch.no_grad():
                activations["layer_0_act_l1"] = torch.mean(torch.abs(out)).item()
                activations["layer_0_act"] = out.clone().detach()

        pre_out = self.fc_2(out)
        if self.log_activations:
            with torch.no_grad():
                activations["layer_1_pre-act_l1"] = torch.mean(torch.abs(pre_out)).item()
                activations["layer_1_pre-act"] = pre_out.clone().detach()
        
        out = self.nonlin(pre_out)
        if self.log_activations:
            with torch.no_grad():
                activations["layer_1_act_l1"] = torch.mean(torch.abs(out)).item()
                activations["layer_1_act"] = out.clone().detach()

        pre_out = self.fc_3(out)
        if self.log_activations:
            with torch.no_grad():
                activations["layer_2_pre-act_l1"] = torch.mean(torch.abs(pre_out)).item()
                activations["layer_2_pre-act"] = pre_out.clone().detach()
        
        out = self.nonlin(pre_out)
        if self.log_activations:
            with torch.no_grad():
                activations["layer_2_act_l1"] = torch.mean(torch.abs(out)).item()
                activations["layer_2_act"] = out.clone().detach()

        pre_out = self.fc_4(out)
        # if self.log_activations:
        #     with torch.no_grad():
        #         activations["layer_3_pre-act_l1"] = torch.mean(torch.abs(pre_out)).item()
        #         activations["layer_3_pre-act"] = pre_out.clone().detach()
        
        out = pre_out #* self.output_mul
        if self.log_activations:
            with torch.no_grad():
                activations["layer_3_logits_l1"] = torch.mean(torch.abs(out)).item()
                activations["layer_3_logits"] = out.clone().detach()
        
        return out, activations




class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, log_activations=True):
        super(SimpleCNN, self).__init__()
        self.log_activations = log_activations
        
        # First convolutional layer with stride=2
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1)
        
        # Second convolutional layer with stride=1
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=16, stride=16)
        
        # Calculate the size of features after convolutions and pooling
        # For a 32x32 input image:
        # After conv1 (stride=2): 16x16
        # After conv2 (stride=1): 16x16
        # After pooling: 8x8
        self.fc_input_size = 32
        
        # Fully connected layer
        self.fc = nn.Linear(self.fc_input_size, num_classes)
        
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        # Initialize conv layers
        ini = VarianceScaling(scale=1.0, mode='fan_in', distribution='truncated_normal')
        ini.initialize(self.conv1.weight)
        ini.initialize(self.conv2.weight)
        
        # Initialize bias terms to zero
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)
        
        # Initialize fully connected layer
        ini.initialize(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    
    def forward(self, x):
        activations = {}  # Dictionary to store activation logs
        
        # First conv layer
        x = self.conv1(x)
        if self.log_activations:
            with torch.no_grad():
                activations["conv1_pre-act_l1"] = torch.mean(torch.abs(x)).item()
                activations["conv1_pre-act"] = x.clone().detach()
        
        x = F.relu(x)
        if self.log_activations:
            with torch.no_grad():
                activations["conv1_act_l1"] = torch.mean(torch.abs(x)).item()
                activations["conv1_act"] = x.clone().detach()
        
        # Second conv layer
        x = self.conv2(x)
        if self.log_activations:
            with torch.no_grad():
                activations["conv2_pre-act_l1"] = torch.mean(torch.abs(x)).item()
                activations["conv2_pre-act"] = x.clone().detach()
        
        x = F.relu(x)
        if self.log_activations:
            with torch.no_grad():
                activations["conv2_act_l1"] = torch.mean(torch.abs(x)).item()
                activations["conv2_act"] = x.clone().detach()
        
        # Max pooling
        x = self.pool(x)
        if self.log_activations:
            with torch.no_grad():
                activations["pool_out_l1"] = torch.mean(torch.abs(x)).item()
                activations["pool_out"] = x.clone().detach()
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layer
        x = self.fc(x)
        if self.log_activations:
            with torch.no_grad():
                activations["fc_out_l1"] = torch.mean(torch.abs(x)).item()
                activations["fc_out"] = x.clone().detach()
        
        return x, activations

def test(
    args, model, device, test_loader, evalmode=True, criterion=F.cross_entropy
):
    if evalmode:
        model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, activations = model(data.view(data.size(0), -1))
            test_loss += criterion(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    return test_loss, correct / len(test_loader.dataset)


if __name__ == '__main__':
    logs = [] 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        os.environ["PYLO_DATA_DIR"]
    except:
        print("ERROR: PYLO_DATA_DIR is not set, please set it to the path of the dataset")
        print("export PYLO_DATA_DIR=$PWD/data")
        exit(0)

    class args(object):
        pass

    args.lr = 1
    args.momentum = 0.9
    args.data_dir = os.environ["PYLO_DATA_DIR"]
    args.batch_size = 4096
    args.no_shuffle = True
    args.epochs = 100
    args.log_interval = 1
    args.num_iters = 5000
    args.width = 1024
    args.wandb_checkpoint_id = 'eb-lab/mup-meta-training/woz3g9l0'
    args.optimizer = 'mup_small_fc_mlp'
    args.num_trials = 5
    args.image_size = 32
    args.dataset_name = 'imagenet'
    args.use_wandb = True
    args.num_classes = 1000
    args.model_testing = False
    args.use_compile = True
    args.test_interval = 500
    args.train_num_workers = 8
    args.test_num_workers = 8
    args.train_prefetch_factor = 10
    args.test_prefetch_factor = 5
    args.use_bf16 = True
    args.use_mixed_precision = args.use_bf16
    IS = args.image_size
    args.log_activations = False
    args.model_type = "mlp"
    args.optimizer = "velo"
    args.group_name = f"mup_small_fc_mlp32_['mumlp-w{args.width}-d3_{args.dataset_name}-{IS}x{IS}x3']_m_mup_final_torch-cuda-fp32-act-paul-cuda"


    # assertions
    if args.model_type == "mumlp" and args.optimizer == "velo":
        raise ValueError("Velo should not be used with MuMLP, use MLP instead")
    if args.model_type in ["cnn", "mlp"] and args.optimizer == "mulo":
        raise ValueError("MuLO should not be used with {}, use MuMLP instead".format(args.model_type))



    if args.model_testing:
        input_width = args.image_size * args.image_size * 3
        model = MuMLP(input_width, width=args.width, num_classes=args.num_classes).to(device)
        # set MuP shapes
        set_base_shapes(model, MuMLP(input_width, width=1, num_classes=args.num_classes), delta=MuMLP(input_width, width=2, num_classes=args.num_classes))

        #set seed before re-init parameters
        set_torch_seed()
        model.reset_parameters()

        # set the biases to have base_width = 1 
        for n,p in model.named_parameters():
            if p.infshape.main != None:
                print(n, p.infshape.main.width_mult(),p.shape,p.infshape.ninf())
                with torch.no_grad():
                    print("{} {} {}".format(p.std(),p.mean(),p.infshape))
                # print(,p.bias.shape)
            # if 'bias' in n:
            #     if p.infshape.main:
            #         p.infshape.main.base_dim = width
        exit(0)
        

    train_loader, test_loader = get_dataset(args)
    train_loader = iter(train_loader)

    if not args.use_bf16:
        torch.set_float32_matmul_precision('high')

    for _ in range(args.num_trials):

        if args.use_wandb:
            run = wandb.init(
                            project='mup-meta-testing',
                            config=dict(wandb_checkpoint_id='eb-lab/mup-meta-training/woz3g9l0',
                                        optimizer='mup_small_fc_mlp'),
                            group=args.group_name
                        )

        input_width = args.image_size * args.image_size * 3
        if args.model_type == "mumlp":
            model = MuMLP(
                in_channels=input_width, 
                width=args.width, 
                num_classes=args.num_classes,
                log_activations=args.log_activations,
                use_mup=True
            ).to(device)# set MuP shapes
            set_base_shapes(
                model, 
                MuMLP(input_width, width=1, num_classes=args.num_classes), 
                delta=MuMLP(input_width, width=2, num_classes=args.num_classes))

            #set seed before re-init parameters
            set_torch_seed()
            model.reset_parameters()
        elif args.model_type == "mlp":
            model = MuMLP(
                in_channels=input_width, 
                width=args.width, 
                num_classes=args.num_classes,
                log_activations=args.log_activations,
                use_mup=False
            ).to(device)
            set_torch_seed()
        elif args.model_type == "cnn":
            model = SimpleCNN(
                num_classes=args.num_classes,
                log_activations=args.log_activations
            ).to(device)
            set_torch_seed()
        else:
            raise ValueError(f"Model type {args.model_type} not supported")
        
        # Compile the model
        if torch.cuda.is_available() and args.use_compile:
            model = torch.compile(model)

        if args.optimizer == "mulo":
            optimizer = MuLO_CUDA(model.parameters(), lr=args.lr)
        elif args.optimizer == "mulo_naive":
            optimizer = MuLO_naive(model.parameters(), lr=args.lr)
        elif args.optimizer == "velo":
            from pylo.optim.Velo import VeLO
            optimizer = VeLO(model.parameters(), lr=args.lr)
        else:
            raise ValueError(f"Optimizer {args.optimizer} not supported")

        model.train()

        pbar = tqdm(
            range(args.num_iters),
            initial=0,
            total=args.num_iters,
            ascii=True,
            desc="Inner Loop",
        )

        # Pre-fetch first batch and start transfer to GPU
        next_data, next_target = next(train_loader)
        if torch.cuda.is_available():
            next_data = next_data.to(device, non_blocking=True)
            next_target = next_target.to(device, non_blocking=True)

        # Create GradScaler for mixed precision training
        scaler = torch.cuda.amp.GradScaler(enabled=args.use_mixed_precision)

        for batch_idx in pbar:
            # Use pre-fetched batch
            data, target = next_data, next_target

            # Overlap data transfer with computation by pre-fetching next batch
            next_data, next_target = next(train_loader)
            if torch.cuda.is_available():
                next_data = next_data.to(device, non_blocking=True)
                next_target = next_target.to(device, non_blocking=True)

            data, target = data.to(device), target.to(device)
            # Flatten data once
            data_flattened = data.view(data.size(0), -1)
            optimizer.zero_grad()

            # Forward pass with autocast
            with torch.cuda.amp.autocast(enabled=args.use_mixed_precision, dtype=torch.bfloat16):
                output, activations = model(data_flattened)
                loss = F.cross_entropy(output, target)

            # Scale loss and call backward
            scaler.scale(loss).backward()

            # Combine operations to reduce memory access
            with torch.no_grad():
                pred = output.argmax(dim=1, keepdim=True)
                accuracy = pred.eq(target.view_as(pred)).sum().item()/data.size(0)

            temp = {
                "iteration": batch_idx,
                "train loss": loss.item(),
                "train_acc": accuracy,
            }

            if args.use_bf16:
                #cast to FP32 so kernel works
                model = model.to(torch.float32)

            if args.log_activations:
                with torch.no_grad():
                    for n,p in model.named_parameters():
                        temp[f"{n}_norm"] = torch.norm(p).item()

            # Unscale gradients and step optimizer
            scaler.unscale_(optimizer)
            if isinstance(optimizer, VeLO):
                optimizer.step(loss)
            else:
                optimizer.step()

            # Update scaler
            scaler.update()

            if batch_idx % args.test_interval == 0:
                test_loss, test_acc = test(args, model, device, test_loader)
                model.train()

                temp.update(
                    {"test_loss":test_loss,
                    "test_acc":test_acc}
                )

            pbar.set_postfix(temp)
            

            if args.use_wandb:
                temp.update(activations)
                run.log(temp)

        if args.use_wandb:
            wandb.finish()

        # del train_loader
        # del test_loader

    exit(0)
