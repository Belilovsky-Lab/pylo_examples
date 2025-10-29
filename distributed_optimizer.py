
# torchrun --standalone --nproc_per_node=8 train_gpt.py
# torchrun --standalone --nproc_per_node=4 train_gpt.py
import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import uuid
import time
import copy
import glob
from dataclasses import dataclass
from functools import lru_cache, partial # Added partial for hook registration
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
torch.empty(1, device="cuda", requires_grad=True).backward() # prevents a bug on some systems
from torch import Tensor, nn
import torch.nn.functional as F
import torch.distributed as dist
from pylo.optim.velo_cuda import VeLO_CUDA
from pylo.optim.AdafacLO_cuda import AdafacLO_CUDA
# use of FlexAttention contributed by @KoszarskyB
from torch.nn.attention.flex_attention import BlockMask, flex_attention
#torch._inductor.config.coordinate_descent_tuning = True # we have banned this flag for new records because it causes compilation to take 30min

# -----------------------------------------------------------------------------
# Custom operators: FP8 matmul by @YouJiacheng

@torch.library.custom_op("nanogpt::mm", mutates_args=())
def mm_op(x: Tensor, w: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[Tensor, Tensor, Tensor]:
    @torch.compile
    def impl(x: Tensor, w: Tensor):
        assert x.is_contiguous() and w.is_contiguous()
        x_f8 = x.div(x_s).to(torch.float8_e4m3fn)
        w_f8 = w.div(w_s).to(torch.float8_e4m3fn)
        out = torch._scaled_mm(
            x_f8,
            w_f8.T,
            out_dtype=torch.bfloat16,
            scale_a=x.new_tensor(x_s, dtype=torch.float32),
            scale_b=x.new_tensor(w_s, dtype=torch.float32),
            use_fast_accum=True,
        )
        return out, x_f8, w_f8

    return impl(x, w)

@mm_op.register_fake
def _(x: Tensor, w: Tensor, *_):
    assert x.ndim == w.ndim == 2
    assert x.shape[1] == w.shape[1]
    assert x.device == w.device
    assert x.is_contiguous() and w.is_contiguous()
    return x @ w.T, x.to(torch.float8_e4m3fn), w.to(torch.float8_e4m3fn)

@torch.library.custom_op("nanogpt::mm_backward", mutates_args=())
def mm_backward_op(g: Tensor, x_f8: Tensor, w_f8: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[Tensor, Tensor]:
    @torch.compile
    def impl(grad: Tensor, x_f8: Tensor, w_f8: Tensor):
        assert grad.is_contiguous()
        x_inv_s = grad.new_tensor(x_s, dtype=torch.float32)
        w_inv_s = grad.new_tensor(w_s, dtype=torch.float32)
        grad_inv_s = grad.new_tensor(grad_s, dtype=torch.float32)
        grad_f8 = grad.div(grad_s).to(torch.float8_e5m2)
        grad_x = torch._scaled_mm(
            grad_f8,
            w_f8.T.contiguous().T,
            out_dtype=torch.bfloat16,
            scale_a=grad_inv_s,
            scale_b=w_inv_s,
            use_fast_accum=False,
        )
        # faster than grad_f8_t @ x_f8, for (d_out, d_in) == (50304, 768)
        grad_w = torch._scaled_mm(
            x_f8.T.contiguous(),
            grad_f8.T.contiguous().T,
            out_dtype=torch.float32,
            scale_a=x_inv_s,
            scale_b=grad_inv_s,
            use_fast_accum=False,
        ).T
        return grad_x, grad_w

    return impl(g, x_f8, w_f8)

@mm_backward_op.register_fake
def _(g: Tensor, x_f8: Tensor, w_f8: Tensor, *_):
    return x_f8.to(torch.bfloat16), w_f8.T.contiguous().T.to(torch.float32)

def backward(ctx, grad_out: Tensor, *_):
    x_f8, w_f8 = ctx.saved_tensors
    x_s, w_s, grad_s = ctx.scales
    grad_x, grad_w = torch.ops.nanogpt.mm_backward(
        grad_out, x_f8, w_f8, x_s, w_s, grad_s
    )
    return grad_x, grad_w, None, None, None

def setup_context(ctx: torch.autograd.function.FunctionCtx, inputs, output):
    *_, x_s, w_s, grad_s = inputs
    _, x_f8, w_f8 = output
    ctx.save_for_backward(x_f8, w_f8)
    ctx.scales = x_s, w_s, grad_s
    ctx.set_materialize_grads(False)

mm_op.register_autograd(backward, setup_context=setup_context)

# -----------------------------------------------------------------------------
# Muon optimizer

@torch.compile
def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X





class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Warning: This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    """
    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95, use_reduce_scatter: bool = True):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, use_reduce_scatter=use_reduce_scatter)
        params = list(params)
        self.use_reduce_scatter = use_reduce_scatter
        
        if use_reduce_scatter:
            # Group by shape for efficient reduce_scatter
            sizes = {p.shape for p in params}
            param_groups = []
            for size in sizes:
                group_params = [p for p in params if p.shape == size]
                param_groups.append(dict(params=group_params))
        else:
            # Simple grouping for all-reduce mode
            param_groups = [dict(params=params)]
        
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        if self.use_reduce_scatter:
            self._step_reduce_scatter()
        else:
            self._step_all_reduce()
    
    def _step_reduce_scatter(self):
        """Reduce scatter version: similar to original Muon implementation."""
        # Efficient systems-wise implementation of step developed by @YouJiacheng,
        # @KonstantinWilleke, @alexrgilbert, @adricarda, @tuttyfrutyee, @vdlad,
        # @ryanyang0, and @vagrawal.
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        reduce_scatter_futures: list[torch.Future] = []
        all_reduce_futures: list[torch.Future] = []
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            grad = torch.empty_like(params[-1])
            grad_pad = [param.grad for param in params] + [torch.zeros_like(params[-1])] * world_size
            for base_i in range(0, len(params), world_size):
                if base_i + rank < len(params):
                    grad = params[base_i + rank].grad
                # This gives strange dynamo warnings
                reduce_scatter_futures.append(dist.reduce_scatter(grad, grad_pad[base_i:base_i + world_size], op=dist.ReduceOp.AVG, async_op=True).get_future())

        idx = 0
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * world_size
            momentum = group["momentum"]
            for base_i in range(0, len(params), world_size):
                reduce_scatter_futures[idx].wait()
                if base_i + rank < len(params):
                    p = params[base_i + rank]
                    grad = p.grad
                    eff_lr = group["lr"] * max(1, p.size(-2) / p.size(-1)) ** 0.5 * getattr(p, "lr_mul", 1.0)
                    eff_weight_decay = group["lr"] * group["weight_decay"] * getattr(p, "wd_mul", 1.0)
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(grad)
                    momentum_buffer = state["momentum_buffer"]
                    p.mul_(1 - eff_weight_decay)
                    momentum_buffer.lerp_(grad, 1 - momentum)
                    grad = grad.lerp_(momentum_buffer, momentum)
                    v = zeropower_via_newtonschulz5(grad.bfloat16(), 5)
                    p.add_(other=v, alpha=-eff_lr)
                idx += 1
                all_reduce_futures.append(dist.all_gather(params_pad[base_i:base_i + world_size], params_pad[base_i + rank], async_op=True).get_future())
        torch.futures.collect_all(all_reduce_futures).wait()
    
    def _step_all_reduce(self):
        """All-reduce version: all-reduce gradients first, then optimize all parameters."""
        all_reduce_futures: list[torch.Future] = []
        
        # All-reduce all gradients
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    all_reduce_futures.append(
                        dist.all_reduce(p.grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                    )
        
        # Wait for all-reduce to complete
        torch.futures.collect_all(all_reduce_futures).wait()
        
        # Optimize all parameters using standard Muon
        for group in self.param_groups:
            momentum = group["momentum"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                    
                grad = p.grad
                eff_lr = group["lr"] * max(1, p.size(-2) / p.size(-1)) ** 0.5 * getattr(p, "lr_mul", 1.0)
                eff_weight_decay = group["lr"] * group["weight_decay"] * getattr(p, "wd_mul", 1.0)
                state = self.state[p]
                
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                
                momentum_buffer = state["momentum_buffer"]
                p.mul_(1 - eff_weight_decay)
                momentum_buffer.lerp_(grad, 1 - momentum)
                grad = grad.lerp_(momentum_buffer, momentum)
                v = zeropower_via_newtonschulz5(grad.bfloat16(), 5)
                p.add_(other=v, alpha=-eff_lr)

class DistAdam(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.01, use_reduce_scatter: bool = True):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, use_reduce_scatter=use_reduce_scatter)
        params = list(params)
        self.use_reduce_scatter = use_reduce_scatter
        
        if use_reduce_scatter:
            # Group by shape for efficient reduce_scatter
            sizes = {p.shape for p in params}
            param_groups = []
            for size in sizes:
                group_params = [p for p in params if p.shape == size]
                param_groups.append(dict(params=group_params))
        else:
            # Simple grouping for all-reduce mode
            param_groups = [dict(params=params)]
        
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        if self.use_reduce_scatter:
            self._step_reduce_scatter()
        else:
            self._step_all_reduce()
    def _step_reduce_scatter(self):
        """Reduce scatter version: similar to original DistAdam implementation."""
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        reduce_scatter_futures: list[torch.Future] = []
        all_reduce_futures: list[torch.Future] = []
        grad_slices = []
        
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            for p in params:
                if p.grad is None:
                    continue
                    
                grad = p.grad
                
                # For 2D tensors (matrices), reduce_scatter along the first dimension
                if grad.ndim >= 2:
                    # Check if first dimension is divisible by world_size
                    if grad.shape[0] % world_size != 0:
                        # Fall back to all-reduce for this parameter
                        all_reduce_futures.append(
                            dist.all_reduce(grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                        )
                        grad_slices.append(None)  # Placeholder
                    else:
                        rank_size = grad.shape[0] // world_size
                        grad_slice = torch.empty_like(grad[:rank_size])
                        reduce_scatter_futures.append(
                            dist.reduce_scatter_tensor(grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                        )
                        grad_slices.append(grad_slice)
                else:
                    # For 1D tensors and scalars, use all-reduce
                    all_reduce_futures.append(
                        dist.all_reduce(grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                    )
                    grad_slices.append(None)  # Placeholder

        # Wait for all communication to complete
        torch.futures.collect_all(reduce_scatter_futures + all_reduce_futures).wait()

        # Now optimize parameters
        reduce_scatter_idx = 0
        all_reduce_idx = 0
        
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            params = group['params']
            
            for p in params:
                if p.grad is None:
                    continue
                    
                lr = group['lr'] * getattr(p, "lr_mul", 1.0)
                state = self.state[p]
                grad = p.grad
                
                # State init
                if not state:
                    state['step'] = torch.tensor(0, dtype=torch.int64, device=p.device)
                
                state['step'] += 1
                t = state['step']
                
                # Handle different parameter types
                if grad.ndim >= 2 and grad.shape[0] % world_size == 0:
                    # Use reduce_scatter result for 2D tensors divisible by world_size
                    rank_size = p.shape[0] // world_size
                    p_slice = p[rank * rank_size:(rank + 1) * rank_size]
                    g_slice = grad_slices[reduce_scatter_idx + all_reduce_idx]
                    
                    if 'exp_avg' not in state:
                        state['exp_avg'] = torch.zeros_like(p_slice)
                        state['exp_avg_sq'] = torch.zeros_like(p_slice)
                    
                    exp_avg = state['exp_avg']
                    exp_avg_sq = state['exp_avg_sq']
                    
                    # weight decay
                    if wd != 0:
                        eff_weight_decay = lr * wd * getattr(p, "wd_mul", 1.0)
                        p_slice.mul_(1 - eff_weight_decay)
                    
                    # update running averages
                    exp_avg.mul_(beta1).add_(g_slice, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(g_slice, g_slice, value=1 - beta2)
                    
                    # bias corrections
                    bias1 = 1 - beta1 ** t
                    bias2 = 1 - beta2 ** t
                    
                    # compute step
                    denom = exp_avg_sq.sqrt().add_(eps)
                    step_size = lr * (torch.sqrt(bias2) / bias1)
                    update = exp_avg.div(denom).mul_(step_size)
                    p_slice.add_(other=update, alpha=-1.0)
                    
                    # All-gather the updated slice back to full parameter
                    all_reduce_futures.append(
                        dist.all_gather_into_tensor(p, p_slice, async_op=True).get_future()
                    )
                    reduce_scatter_idx += 1
                else:
                    # Use all-reduce result for other tensors
                    if 'exp_avg' not in state:
                        state['exp_avg'] = torch.zeros_like(p)
                        state['exp_avg_sq'] = torch.zeros_like(p)
                    
                    exp_avg = state['exp_avg']
                    exp_avg_sq = state['exp_avg_sq']
                    
                    # weight decay
                    if wd != 0:
                        eff_weight_decay = lr * wd * getattr(p, "wd_mul", 1.0)
                        p.mul_(1 - eff_weight_decay)
                    
                    # update running averages
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    
                    # bias corrections
                    bias1 = 1 - beta1 ** t
                    bias2 = 1 - beta2 ** t
                    
                    # compute step
                    denom = exp_avg_sq.sqrt().add_(eps)
                    step_size = lr * (torch.sqrt(bias2) / bias1)
                    update = exp_avg.div(denom).mul_(step_size)
                    p.add_(other=update, alpha=-1.0)
                    all_reduce_idx += 1

        # Wait for all all-gather operations to complete
        torch.futures.collect_all(all_reduce_futures).wait()
    def _step_all_reduce(self):
        """All-reduce version: all-reduce gradients first, then optimize all parameters."""
        all_reduce_futures: list[torch.Future] = []
        
        # All-reduce all gradients
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    all_reduce_futures.append(
                        dist.all_reduce(p.grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                    )
        
        # Wait for all-reduce to complete
        torch.futures.collect_all(all_reduce_futures).wait()
        
        # Optimize all parameters using standard Adam
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            params = group['params']
            
            for p in params:
                if p.grad is None:
                    continue
                    
                lr = group['lr'] * getattr(p, "lr_mul", 1.0)
                state = self.state[p]
                grad = p.grad
                
                # State init
                if not state:
                    state['step'] = torch.tensor(0, dtype=torch.int64, device=p.device)
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                state['step'] += 1
                t = state['step']
                
                # weight decay
                if wd != 0:
                    eff_weight_decay = lr * wd * getattr(p, "wd_mul", 1.0)
                    p.mul_(1 - eff_weight_decay)
                
                # update running averages
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # bias corrections
                bias1 = 1 - beta1 ** t
                bias2 = 1 - beta2 ** t
                
                # compute step
                denom = exp_avg_sq.sqrt().add_(eps)
                step_size = lr * (torch.sqrt(bias2) / bias1)
                update = exp_avg.div(denom).mul_(step_size)
                p.add_(other=update, alpha=-1.0)


class DistLopt(torch.optim.Optimizer):
    """
    Distributed wrapper for learned optimizers.
    
    Supports two communication patterns:
    (a) use_reduce_scatter=True: Similar to Muon - reduce_scatter gradients, optimize local chunks, all_gather parameters
        - For VeLO_CUDA: Runs RNN per-parameter (different from standard VeLO which runs RNN once for all params)
    (b) use_reduce_scatter=False: All-reduce gradients first, then optimize
        - For VeLO_CUDA: Runs RNN once for all parameters (standard VeLO behavior)
    """
    def __init__(self, 
        params,
        use_reduce_scatter: bool = True, 
        lr: float = 1.0, 
        num_iterations: int = 1300,
        lopt_type: str = "AdafacLO_CUDA",
        **lopt_kwargs):
        params = list(params)
        self.use_reduce_scatter = use_reduce_scatter
        self.lopt_type = lopt_type  # Store lopt_type for later use
        
        if use_reduce_scatter:
            # Group by shape like Muon for efficient reduce_scatter
            sizes = {p.shape for p in params}
            param_groups = []
            for size in sizes:
                group_params = [p for p in params if p.shape == size]
                param_groups.append(dict(params=group_params))
        else:
            # Simple grouping for all-reduce mode
            param_groups = [dict(params=params)]
        
        # Include lr and use_reduce_scatter in defaults so they appear in param_groups
        defaults = dict(lr=lr, use_reduce_scatter=use_reduce_scatter)
        super().__init__(param_groups, defaults)
        
        # Create the base learned optimizer
        if lopt_type == "AdafacLO_CUDA":
            self.lopt = AdafacLO_CUDA(params, lr=lr, **lopt_kwargs)
        elif lopt_type == "VeLO_CUDA":
            self.lopt = VeLO_CUDA(params, num_steps=num_iterations, lr=lr, legacy=False)
        else:
            raise ValueError(f"Invalid lopt_type: {lopt_type}")
        
        # Create parameter to group mapping for efficient lookup
        self.param_to_group = {}
        for group in self.lopt.param_groups:
            for p in group["params"]:
                self.param_to_group[p] = group
    
    @torch.no_grad()
    def step(self, loss=None):
        if self.use_reduce_scatter:
            self._step_reduce_scatter(loss)
        else:
            self._step_all_reduce(loss)
    
    def _step_reduce_scatter(self, loss=None):
        """Reduce scatter version: handle communication like Muon, use learned optimizer for optimization.
        
        For VeLO_CUDA: Runs RNN per-parameter after reduce_scatter (each rank processes its params independently).
        For AdafacLO_CUDA: Uses single tensor step function.
        """
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        reduce_scatter_futures: list[torch.Future] = []
        all_reduce_futures: list[torch.Future] = []
        
        # For VeLO_CUDA: Update loss buffer and increment step counter
        to_lstm_from_loss = None
        if self.lopt_type == "VeLO_CUDA":
            if loss is None:
                raise ValueError("VeLO_CUDA requires loss to be passed to step()")
            # Update loss buffer
            self.lopt.loss_buffer = self.lopt.buffer_loss_fns.update(self.lopt.loss_buffer, loss)
            to_lstm_from_loss = self.lopt.buffer_loss_fns.features(self.lopt.loss_buffer)
            # Increment step counter
            for group in self.lopt.param_groups:
                group["step"] += 1
        
        # Phase 1: Reduce scatter gradients
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            grad = torch.empty_like(params[-1])
            grad_pad = [param.grad for param in params] + [torch.zeros_like(params[-1])] * world_size
            for base_i in range(0, len(params), world_size):
                if base_i + rank < len(params):
                    grad = params[base_i + rank].grad
                reduce_scatter_futures.append(
                    dist.reduce_scatter(grad, grad_pad[base_i:base_i + world_size], 
                                      op=dist.ReduceOp.AVG, async_op=True).get_future()
                )
        
        # Phase 2: Optimize local parameters using the learned optimizer
        idx = 0
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * world_size
            for base_i in range(0, len(params), world_size):
                reduce_scatter_futures[idx].wait()
                if base_i + rank < len(params):
                    p = params[base_i + rank]
                    # Process this single parameter using the appropriate learned optimizer
                    if self.lopt_type == "VeLO_CUDA":
                        # Run RNN for this single parameter
                        self._step_single_param_velo(p, to_lstm_from_loss)
                    else:
                        self._step_single_param_adafac(p)
                idx += 1
                # Phase 3: All gather updated parameters
                all_reduce_futures.append(
                    dist.all_gather(params_pad[base_i:base_i + world_size], 
                                  params_pad[base_i + rank], async_op=True).get_future()
                )
        
        torch.futures.collect_all(all_reduce_futures).wait()
    
    def _step_single_param_adafac(self, param: Tensor):
        """Step a single parameter using AdafacLO_CUDA internal logic."""
        if param.grad is None:
            return
        
        # Get optimizer state for this parameter
        state = self.lopt.state[param]
        
        # Initialize state if needed (same as AdafacLO_CUDA)
        if len(state) == 0:
            from pylo.optim.AdafacLO_cuda import _get_scalar_dtype, _factored_dims
            state["step"] = torch.tensor(0.0, dtype=_get_scalar_dtype())
            
            shape = param.grad.shape
            factored_dims = _factored_dims(shape, factored=True, min_dim_size_to_factor=1)
            
            if factored_dims is not None:
                dc, dr = factored_dims
                row_shape = list(param.grad.shape)
                row_shape[dr] = 1
                col_shape = list(param.grad.shape)
                col_shape[dc] = 1
                state["exp_avg_sq_r"] = param.grad.new_zeros([3] + row_shape)
                state["exp_avg_sq_c"] = param.grad.new_zeros([3] + col_shape)
                state["exp_avg_sq"] = torch.zeros_like(param.grad, memory_format=torch.preserve_format)
            else:
                state["exp_avg_sq_r"] = param.grad.new_zeros((3,) + param.grad.shape)
                state["exp_avg_sq_c"] = param.grad.new_zeros((3,) + param.grad.shape)
                state["exp_avg_sq"] = torch.zeros_like(param.grad, memory_format=torch.preserve_format)
            
            state["exp_avg"] = param.grad.new_zeros((3,) + param.grad.shape)
        
        # Get optimizer group settings (use first group from lopt)
        lopt_group = self.lopt.param_groups[0]
        
        # Import and call the single tensor step function
        from pylo.optim.AdafacLO_cuda import _single_tensor_adafactor
        
        _single_tensor_adafactor(
            self=self.lopt,
            params=[param],
            grads=[param.grad],
            exp_avg_sq_rs=[state.get("exp_avg_sq_r", None)],
            exp_avg_sq_cs=[state.get("exp_avg_sq_c", None)],
            exp_avg_sqs=[state.get("exp_avg_sq", None)],
            exp_avgs=[state.get("exp_avg", None)],
            state_steps=[state["step"]],
            eps=lopt_group["eps"],
            lr=lopt_group["lr"],
            step_mult=self.lopt.step_mult,
            exp_mult=self.lopt.exp_mult,
            weight_decay=lopt_group["weight_decay"],
            momentum_dtype=lopt_group["momentum_dtype"],
            clipping_threshold=lopt_group["clipping_threshold"],
            unscaled_wd=lopt_group["unscaled_wd"],
        )
    
    def _step_single_param_velo(self, param: Tensor, to_lstm_from_loss):
        """Step a single parameter using VeLO_CUDA by running RNN for just this parameter."""
        if param.grad is None:
            return
        
        from pylo.optim.velo_cuda import fractional_tanh_embed, lstm_features_for_tensor, factored_dims, safe_rsqrt
        import velo_cuda_kernel
        
        grad = torch.clip(param.grad, -1000.0, 1000.0)
        state = self.lopt.state[param]
        layer_idx = state["layer_idx"]
        
        # Get group for this parameter using precomputed mapping
        group = self.param_to_group[param]
        fraction_trained = group["step"] / self.lopt.num_steps
        exp_mult = group["exp_mult"]
        step_mult = group["step_mult"]
        weight_decay = group["weight_decay"]
        lr = group["lr"]
        beta_m = group["initial_momentum_decays"]
        beta_rms = group["initial_rms_decays"]
        beta_adafactor = group["initial_adafactor_decays"]
        
        # Get state variables
        mom = state["mom"]
        rms = state["rms"]
        p_shape = param.shape
        
        # Update momentum and RMS (same as VeLO's _step_loop)
        batch_g = grad[None, ...]
        beta_m_view = beta_m.view(-1, *[1] * len(p_shape))
        beta_rms_view = beta_rms.view(-1, *[1] * len(p_shape))
        beta_adafactor_view = beta_adafactor.view(-1, *[1] * len(p_shape))
        
        mom.lerp_(batch_g, 1 - beta_m_view.to(grad.dtype))
        rms.lerp_(batch_g**2, 1 - beta_rms_view.to(grad.dtype))
        
        # Update factored accumulators
        f_dims = factored_dims(p_shape)
        grad_sqr = torch.square(grad) + 1e-30
        if f_dims is not None:
            dc, dr = f_dims
            state["fac_vec_row"].lerp_(
                grad_sqr.mean(dim=dr, keepdim=True)[None, ...],
                1 - beta_adafactor_view.to(state["fac_vec_row"].dtype),
            )
            state["fac_vec_col"].lerp_(
                grad_sqr.mean(dim=dc, keepdim=True)[None, ...],
                1 - beta_adafactor_view.to(state["fac_vec_col"].dtype),
            )
        else:
            state["fac_vec_v"].lerp_(
                grad_sqr[None, ...],
                1 - beta_adafactor_view.to(state["fac_vec_v"].dtype),
            )
        # Compute features for this single parameter
        fraction_left = fractional_tanh_embed(fraction_trained).to(self.lopt.device)
        rnn_input = lstm_features_for_tensor(
            param, grad, mom, rms,
            fraction_left, to_lstm_from_loss, self.lopt.device
        )
        
        # Run RNN for this single parameter (batch size = 1)
        rnn_input = rnn_input.unsqueeze(0)  # Add sequence dimension: [1, features]
        rnn_input = torch.flip(rnn_input, [0])
        
        # Extract hidden state for this specific layer
        current_hidden = (
            self.lopt.lstm_hidden_state[0][layer_idx:layer_idx+1],  # [1, hidden_size]
            self.lopt.lstm_hidden_state[1][layer_idx:layer_idx+1]   # [1, hidden_size]
        )
        
        control_params, lr_mult, new_hidden = self.lopt.rnn(rnn_input, current_hidden)
        
        # Update the hidden state for this specific layer
        self.lopt.lstm_hidden_state[0][layer_idx] = new_hidden[0][0]
        self.lopt.lstm_hidden_state[1][layer_idx] = new_hidden[1][0]
        
        # Extract control params and lr_mult for this parameter
        control_param = control_params[0]  # [features]
        lr_mult_val = lr_mult[0]  # scalar
        
        # Use kernel processing (same as VeLO's _process_param_kernel)
        if not hasattr(self.lopt, 'second_moment'):
            self.lopt.second_moment = torch.zeros(30, dtype=torch.float32, device=self.lopt.device)
        self.lopt.second_moment.zero_()
        
        # Get MLP weights
        self.lopt.network_stack.update_params(control_param)
        mlp_params = dict(self.lopt.network_stack.named_parameters())
        
        # Calculate row/col factors
        if f_dims is not None:
            d1, d0 = f_dims
            dc, dr = d1, d0
            vector_like = 0
            fac_vec_row = state["fac_vec_row"]
            fac_vec_col = state["fac_vec_col"]
            
            reduced_d1 = (d1 + 1) - 1 if (d1 + 1) > (d0 + 1) else (d1 + 1)
            row_col_mean = torch.mean(fac_vec_row, dim=reduced_d1, keepdim=True)
            row_factor = safe_rsqrt(fac_vec_row / (row_col_mean + 1e-9))
            col_factor = safe_rsqrt(fac_vec_col)
        else:
            dc, dr = 0, 0
            vector_like = 1
            fac_vec_row = state["fac_vec_v"]
            fac_vec_col = state["fac_vec_v"]
            row_factor = safe_rsqrt(fac_vec_row + 1e-9)
            col_factor = torch.ones_like(row_factor)
        
        # Debug output
        # print(f"DEBUG: param shape: {param.shape}, grad shape: {grad.shape}")
        # print(f"DEBUG: mom shape: {mom.shape}, rms shape: {rms.shape}")
        # print(f"DEBUG: row_factor shape: {row_factor.shape}, col_factor shape: {col_factor.shape}")
        # print(f"DEBUG: fac_vec_row shape: {fac_vec_row.shape}, fac_vec_col shape: {fac_vec_col.shape}")
        # print(f"DEBUG: f_dims: {f_dims}, dc: {dc}, dr: {dr}, vector_like: {vector_like}")
        
        # Call CUDA kernel
        velo_cuda_kernel.velo_kernel_simple(
            grad, param, mom, rms,
            row_factor, col_factor,
            fac_vec_row, fac_vec_col,
            self.lopt.second_moment,
            mlp_params["input_weights"].data, mlp_params["input_bias"].data,
            mlp_params["hidden_weights.0"].data, mlp_params["hidden_bias.0"].data,
            mlp_params["output_weights"].data, mlp_params["output_bias"].data,
            lr * lr_mult_val.item(),
            step_mult, exp_mult, 1e-6, weight_decay,
            dc, dr, vector_like,
        )
    
    def _step_all_reduce(self, loss=None):
        """All-reduce version: all-reduce gradients first, then call optimizer.step()."""
        all_reduce_futures: list[torch.Future] = []
        
        # All-reduce all gradients
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    all_reduce_futures.append(
                        dist.all_reduce(p.grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                    )
        
        # Wait for all-reduce to complete
        torch.futures.collect_all(all_reduce_futures).wait()
        
        # Call the base optimizer step (pass loss if using VeLO_CUDA)
        if self.lopt_type == "VeLO_CUDA":
            if loss is None:
                raise ValueError("VeLO_CUDA requires loss to be passed to step()")
            self.lopt.step(loss)
        else:
            self.lopt.step()

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the model

def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))

class CastedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, use_fp8=False, x_s=1.0, w_s=1.0, grad_s=1.0):
        super().__init__(in_features, out_features, bias=False)
        self.use_fp8 = use_fp8
        self.x_s = x_s
        self.w_s = w_s
        self.grad_s = grad_s

    def reset_parameters(self) -> None:
        std = 0.5 * (self.in_features ** -0.5) # 0.5 is a bit better than the default 1/sqrt(3)
        bound = (3 ** 0.5) * std
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)

    def forward(self, x: Tensor):
        if self.use_fp8 and self.training:
            _x = x.flatten(0, -2)
            out: Tensor = torch.ops.nanogpt.mm(_x, self.weight, x_s=self.x_s, w_s=self.w_s, grad_s=self.grad_s)[0]
            return out.reshape(*x.shape[:-1], -1)
        else:
            return F.linear(x, self.weight.type_as(x))

class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        # half-truncate RoPE by @YouJiacheng (w/ base freq tuning)
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim//4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.cos = nn.Buffer(theta.cos(), persistent=False)
        self.sin = nn.Buffer(theta.sin(), persistent=False)

    def forward(self, x_BTHD: Tensor):
        assert self.cos.size(0) >= x_BTHD.size(-3)
        cos, sin = self.cos[None, :x_BTHD.size(-3), None, :], self.sin[None, :x_BTHD.size(-3), None, :]
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, head_dim=128):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        hdim = num_heads * head_dim
        std = 0.5 * (dim ** -0.5)
        bound = (3 ** 0.5) * std # improved init scale by @YouJiacheng
        # merged QKV weights: suggested by many, implemented by @fernbear.bsky.social, and further improved by @YouJiacheng
        # https://x.com/hi_tysam/status/1879699187107033311
        self.qkv_w = nn.Parameter(torch.empty(3, hdim, dim).uniform_(-bound, bound))
        self.rotary = Rotary(head_dim, max_seq_len)
        self.c_proj = CastedLinear(hdim, dim)
        self.c_proj.weight.detach().zero_() # zero init suggested by @Grad62304977
        # scale the attention logits by given constant, instead of the default head_dim**-0.5, by @leloykun
        # inspired by learnable scalars used by @brendanh0gan https://x.com/hi_tysam/status/1879693583898591283
        self.attn_scale = 0.12

    def forward(self, x: Tensor, ve: Tensor | None, lambdas: Tensor, block_mask: BlockMask):
        B, T = x.size(0), x.size(1) # batch size, sequence length
        assert B == 1, "Must use batch size = 1 for FlexAttention"
        q, k, v = F.linear(x, self.qkv_w.flatten(end_dim=1).type_as(x)).view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)
        q, k = norm(q), norm(k) # QK norm @Grad62304977
        q, k = self.rotary(q), self.rotary(k)
        if ve is not None:
            v = lambdas[0] * v + lambdas[1] * ve.view_as(v) # @KoszarskyB & @Grad62304977
        else: # skip mid-layers token value embeddings by @YouJiacheng
            v = lambdas[0] * v
        y = flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=block_mask, scale=self.attn_scale).transpose(1, 2)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.c_fc = CastedLinear(dim, hdim)
        self.c_proj = CastedLinear(hdim, dim)
        self.c_proj.weight.detach().zero_() # zero init suggested by @Grad62304977

    def forward(self, x: Tensor):
        x = self.c_fc(x)
        x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, layer_idx: int):
        super().__init__()
        # skip attention of blocks.7 (the 8th layer) by @YouJiacheng
        self.attn = CausalSelfAttention(dim, num_heads, max_seq_len) if layer_idx != 7 else None
        self.mlp = MLP(dim)

    def forward(self, x: Tensor, ve: Tensor | None, x0: Tensor, lambdas: Tensor, sa_lambdas: Tensor, block_mask: BlockMask):
        x = lambdas[0] * x + lambdas[1] * x0
        if self.attn is not None:
            x = x + self.attn(norm(x), ve, sa_lambdas, block_mask)
        x = x + self.mlp(norm(x))
        return x

# -----------------------------------------------------------------------------
# The main model

def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, model_dim: int, max_seq_len: int):
        super().__init__()
        vocab_size = next_multiple_of_n(vocab_size, n=128)
        self.embed = nn.Embedding(vocab_size, model_dim)
        # token value embeddings by @KoszarskyB - inspired by @Grad62304977's value residual implementation following https://arxiv.org/abs/2410.17897
        # value embedding code simplification inspired by @ragulpr https://github.com/KellerJordan/modded-nanogpt/pull/78
        self.value_embeds = nn.ModuleList([nn.Embedding(vocab_size, model_dim) for _ in range(3)])
        self.blocks = nn.ModuleList([Block(model_dim, num_heads, max_seq_len, i) for i in range(num_layers)])
        # there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency.
        # suggested to me by @Grad62304977. this originates from Karpathy's experiments.
        self.lm_head = CastedLinear(model_dim, vocab_size, use_fp8=True, x_s=(model_dim**0.5)/448, w_s=24/448, grad_s=1/448)
        self.lm_head.weight.detach().zero_() # @Grad62304977
        # Add learnable skip connection weights for decoder layers
        assert num_layers % 2 == 0
        pad = (-num_layers * 5) % dist.get_world_size()
        self.scalars = nn.Parameter(torch.cat([
            torch.ones(num_layers), # skip_weights
            *[torch.tensor([1.0, 0.0]) for _ in range(num_layers)], # block lambdas
            *[torch.tensor([0.5, 0.5]) for _ in range(num_layers)], # SA lambdas
            torch.ones(pad),
        ]))
        # set learning rates
        for param in self.embed.parameters():
            param.lr_mul = 75.
        for param in self.value_embeds.parameters():
            param.lr_mul = 75.
        self.lm_head.weight.lr_mul = 27.5
        self.scalars.lr_mul = 5.0

    def create_blockmasks(self, input_seq: Tensor, sliding_window_num_blocks: Tensor):
        BLOCK_SIZE = 128
        docs = (input_seq == 50256).cumsum(0)

        def document_causal(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            document_mask = docs[q_idx] == docs[kv_idx]
            return causal_mask & document_mask

        def dense_to_ordered(dense_blockmask: Tensor):
            num_blocks = dense_blockmask.sum(dim=-1, dtype=torch.int32)
            indices = dense_blockmask.argsort(dim=-1, descending=False, stable=True).flip(-1).to(torch.int32)
            return num_blocks[None, None].contiguous(), indices[None, None].contiguous()

        # manual block mask creation by @YouJiacheng
        assert len(input_seq) % BLOCK_SIZE == 0
        NUM_BLOCKS = len(input_seq) // BLOCK_SIZE
        block_idx = torch.arange(NUM_BLOCKS, dtype=torch.int32, device="cuda")
        causal_blockmask_any = block_idx[:, None] >= block_idx
        causal_blockmask_all = block_idx[:, None] > block_idx
        docs_low = docs.view(-1, BLOCK_SIZE)[:, 0].contiguous()
        docs_high = docs.view(-1, BLOCK_SIZE)[:, -1].contiguous()
        document_blockmask_any = (docs_low[:, None] <= docs_high) & (docs_high[:, None] >= docs_low)
        document_blockmask_all = (docs_low[:, None] == docs_high) & (docs_high[:, None] == docs_low)
        blockmask_any = causal_blockmask_any & document_blockmask_any
        blockmask_all = causal_blockmask_all & document_blockmask_all
        partial_kv_num_blocks, partial_kv_indices = dense_to_ordered(blockmask_any & ~blockmask_all)
        full_kv_num_blocks, full_kv_indices = dense_to_ordered(blockmask_all)
        def build_bm(window_size_blocks: Tensor) -> BlockMask:
            return BlockMask.from_kv_blocks(
                torch.clamp_max(partial_kv_num_blocks, torch.clamp_min(window_size_blocks - full_kv_num_blocks, 1)),
                partial_kv_indices,
                torch.clamp_max(full_kv_num_blocks, window_size_blocks - 1),
                full_kv_indices,
                BLOCK_SIZE=BLOCK_SIZE,
                mask_mod=document_causal,
            )
        # Long-short SWA block masks by @leloykun & @YouJiacheng, adapated from suggestion by @Grad62304977, following Gemma 2 paper
        return build_bm(sliding_window_num_blocks), build_bm(sliding_window_num_blocks // 2)

    def forward(self, input_seq: Tensor, target_seq: Tensor, sliding_window_num_blocks: Tensor):
        assert input_seq.ndim == 1

        ve = [value_embed(input_seq) for value_embed in self.value_embeds]
        # 012 ... 012 structure on token value embeddings by @YouJiacheng, improved on @leloykun's U-net structure
        ve = [ve[0], ve[1], ve[2]] + [None] * (len(self.blocks) - 6) + [ve[0], ve[1], ve[2]]
        assert len(ve) == len(self.blocks)

        long_bm, short_bm = self.create_blockmasks(input_seq, sliding_window_num_blocks)
        block_masks = [long_bm, short_bm, short_bm, short_bm, long_bm, short_bm, short_bm, long_bm, short_bm, short_bm, short_bm, long_bm]
        assert len(block_masks) == len(self.blocks)

        x = x0 = norm(self.embed(input_seq)[None]) # use of norm here by @Grad62304977

        # U-net design by @brendanh0gan
        skip_connections = []
        skip_weights = self.scalars[:(len(self.blocks) // 2)]
        lambdas = self.scalars[1 * len(self.blocks): 3 * len(self.blocks)].view(-1, 2)
        sa_lambdas = self.scalars[3 * len(self.blocks): 5 * len(self.blocks)].view(-1, 2)

        n = len(self.blocks) // 2

        for i in range(len(self.blocks)):
            if i >= n:
                x = x + skip_weights[i - n] * skip_connections.pop()
            x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
            if i < n:
                skip_connections.append(x)

        x = norm(x)
        logits = self.lm_head(x).float()
        # @Grad62304977 added tanh softcapping following Gemma 2 paper, @KoszarskyB reduced it from 30 to 15, @YouJiacheng shifted it by +15 (2*sigmoid(2*x)=tanh(x)+1)
        logits = 30 * torch.sigmoid(logits / (7.5 * x.size(-1)**0.5))
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_seq, reduction="sum" if self.training else "mean")
        return loss

# -----------------------------------------------------------------------------
# Distributed data loader

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True) # avoid pin_memory copy by @YouJiacheng
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy by @YouJiacheng
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

# find world_size starting indicies, such that each begins with token 50256 and local_batches don't overlap
def find_batch_starts(tokens: Tensor, pos: int, local_batch_size: int, max_batch_span: int):
    boundary_mask = tokens[pos : pos + max_batch_span] == 50256
    boundary_positions = torch.nonzero(boundary_mask, as_tuple=False).squeeze(-1) + pos
    start = boundary_positions[0].item()
    starts = []
    for i in range(1, len(boundary_positions)):
        end = boundary_positions[i].item() 
        if end - start >= local_batch_size:
            starts.append(start) # append start once end pos is confirmed
            if len(starts) == dist.get_world_size():
                return starts, end - pos
            start = end
    assert False # increase max_batch_span if necessary

def distributed_data_generator(filename_pattern: str, batch_size: int, align_to_bos: bool):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
    assert batch_size % world_size == 0
    local_batch_size = batch_size // world_size
    file_iter = iter(files) # use itertools.cycle(files) instead if you want to do multi-epoch training
    tokens, pos = _load_data_shard(next(file_iter)), 0
    max_batch_span = 2 * batch_size if align_to_bos else batch_size # provide buffer to handle samples up to length local_batch_size
    while True:
        if pos + max_batch_span + 1 >= len(tokens):
            tokens, pos = _load_data_shard(next(file_iter)), 0
        if align_to_bos:
            batch_starts, batch_span = find_batch_starts(tokens, pos, local_batch_size, max_batch_span)
            start_idx = batch_starts[rank]
        else:
            batch_span = batch_size
            start_idx = pos + rank * local_batch_size
        buf = tokens[start_idx:][:local_batch_size + 1]
        inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True) # no sync on host side;
        targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True) # H2D in another stream isn't helpful.
        pos += batch_span
        yield inputs, targets

# -----------------------------------------------------------------------------
# int main


import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Train GPT model with different optimizers")
    parser.add_argument("--optimizer", type=str, default="muon", 
                       help="Optimizer to use (e.g., 'muon', 'velo_all', 'velo_hidden')")
    parser.add_argument("--reduce_scatter", action="store_true", 
                       help="Enable reduce scatter optimization")
    return parser.parse_args()

# Parse command line arguments
parsed_args = parse_args()


@dataclass
class Hyperparameters:
    # data
    train_files = "data/fineweb10B/fineweb_train_*.bin" # input .bin to train on
    val_files = "data/fineweb10B/fineweb_val_*.bin" # input .bin to eval validation loss on
    val_tokens = 10485760 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    train_seq_len = 48*1024 # FlexAttention sequence length
    val_seq_len = 4*64*1024 # FlexAttention sequence length for validation
    # optimization
    num_iterations = 100 # number of iterations to run
    cooldown_frac = 0.45 # fraction of training spent cooling down the learning rate
    # optimizer selection: "muon", "velo_all", "velo_hidden"
    optimizer_mode = parsed_args.optimizer # "muon": use Muon for hidden weights, "velo_all": use VeLO_CUDA for all weights, "velo_hidden": use VeLO_CUDA only for hidden weights
    # scheduler configuration: enable/disable LR scheduling per optimizer
    use_scheduler_opt1 = True # apply LR scheduler to first optimizer (DistAdam/VeLO_CUDA)
    use_scheduler_opt2 = True # apply LR scheduler to second optimizer (Muon/VeLO_CUDA) - only used in muon/velo_hidden modes
    use_cosine_lr = True # use cosine LR scheduler for first optimizer (DistAdam/VeLO_CUDA)\
    overlapped_allreduce = True
    reduce_scatter = parsed_args.reduce_scatter
    # evaluation and logging
    val_loss_every = 125 # every how many steps to evaluate val loss? 0 for only at the end
    save_checkpoint = False
    
args = Hyperparameters()

# torchrun sets these env variables
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
if world_size != 8:
    print(f"Warning: This code is designed for 8xH100, but running with world_size={world_size}")
assert torch.cuda.is_available()
device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)
dist.init_process_group(backend="nccl", device_id=device)
dist.barrier()
master_process = (rank == 0) # this process will do logging, checkpointing etc.

# begin logging
logfile = None
if master_process:
    run_id = uuid.uuid4()
    os.makedirs("logs", exist_ok=True)
    if parsed_args.reduce_scatter:
        reduce_scatter_suffix = "_reduce_scatter"
    else:
        reduce_scatter_suffix = ""
    logfile = f"logs/{run_id}_{parsed_args.optimizer}{reduce_scatter_suffix}.txt"
    print(logfile)
def print0(s, console=False):
    if master_process:
        with open(logfile, "a") as f:
            if console:
                print(s)
            print(s, file=f)

# begin by printing this file (the Python code)
print0(code)
print0("="*100)
# log information about the hardware/software environment this is running on
print0(f"Running Python {sys.version}")
print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}")
def nvidia_smi():
    import subprocess  # avoid top level import
    return subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout
print0(nvidia_smi())
print0("="*100)

model: nn.Module = GPT(vocab_size=50257, num_layers=12, num_heads=6, model_dim=768, max_seq_len=max(args.train_seq_len, args.val_seq_len)).cuda()
for m in model.modules():
    if isinstance(m, nn.Embedding):
        m.bfloat16()
for param in model.parameters():
    dist.broadcast(param.detach(), 0)

# collect the parameters to optimize
hidden_matrix_params = [p for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n]
embed_params = [p for n, p in model.named_parameters() if "embed" in n]
scalar_params = [p for p in model.parameters() if p.ndim < 2]
head_params = [model.lm_head.weight]

# init the optimizer(s)
# small adam epsilon by @YouJiacheng. this is an alternate method of fixing the world_size dependence
# discovered by @fernbear.bsky.social https://x.com/hi_tysam/status/1879692937589875094
if args.optimizer_mode == "muon":
    # Original setup: Muon for hidden weights, DistAdam for others
    optimizer1 = DistAdam(scalar_params + head_params + embed_params, lr=0.008, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0,
    use_reduce_scatter=True,)
    optimizer2 = Muon(hidden_matrix_params, lr=0.05, momentum=0.95, weight_decay=0,
                         use_reduce_scatter=args.reduce_scatter,  )# or False for all-reduce mode)
    optimizers = [optimizer1, optimizer2]
elif args.optimizer_mode == "adam_all":
    optimizer1 = DistAdam(scalar_params + head_params + embed_params, lr=0.008, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0, use_reduce_scatter=args.reduce_scatter)
    optimizer2 = DistAdam(hidden_matrix_params,lr=0.008, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0, use_reduce_scatter=args.reduce_scatter)
    optimizers = [optimizer1, optimizer2]

elif args.optimizer_mode == "velo":
    
    optimizer1 = DistAdam(scalar_params + head_params + embed_params, lr=0.008, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0,
    use_reduce_scatter=True,)
    optimizer2 = DistLopt(hidden_matrix_params, 
                         use_reduce_scatter=args.reduce_scatter,  # or False for all-reduce mode
                         use_overlapped_allreduce=args.overlapped_allreduce,
                         lr=1.0, 
                         num_iterations=args.num_iterations,
                         lopt_type="VeLO_CUDA",)
    optimizers = [optimizer1, optimizer2]

elif args.optimizer_mode == "adafac_lo":
    
    optimizer1 = DistAdam(scalar_params + head_params + embed_params, lr=0.008, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0,
    use_reduce_scatter=True,)
    optimizer2 = DistLopt(hidden_matrix_params,
                         use_reduce_scatter=args.reduce_scatter,
                         lr=1.0, 
                         num_iterations=args.num_iterations,
                         lopt_type="AdafacLO_CUDA",)
    optimizers = [optimizer1, optimizer2]
else:
    raise ValueError(f"Unknown optimizer_mode: {args.optimizer_mode}")

for opt in optimizers:
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]

# learning rate schedule: stable then decay
def get_lr(step: int):
    x = step / args.num_iterations # progress in training
    assert 0 <= x < 1
    if x < 1 - args.cooldown_frac:
        return 1.0
    else:
        w = (1 - x) / args.cooldown_frac
        return w * 1.0 + (1 - w) * 0.1

def get_lr_cosine(step: int):
    import math
    x = step / args.num_iterations # progress in training
    assert 0 <= x < 1
    # Cosine annealing from 1.0 to 0.1
    return 0.1 + 0.5 * (1.0 - 0.1) * (1 + math.cos(math.pi * x))

# attention window size schedule: linearly increase
@lru_cache(1)
def get_window_size_blocks_helper(window_size: int):
    return torch.tensor(window_size // 128, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
def get_window_size_blocks(step: int):
    x = step / args.num_iterations # progress in training
    assert 0 <= x <= 1
    # Linearly increase the block-wise sliding window size over training 128 -> 1792
    # increase by @fernbear.bsky.social; block-wise by @YouJiacheng
    window_size = next_multiple_of_n(1728 * x, n=128)
    return get_window_size_blocks_helper(window_size)

model: nn.Module = torch.compile(model, dynamic=False)



########################################
#      Overlap Communication Setup     #
########################################

if args.overlapped_allreduce:

    # Create parameter buckets for better overlap
    def create_buckets(params, bucket_size_mb=25):
        """Group parameters into buckets of approximately bucket_size_mb MB each"""
        buckets = []
        current_bucket = []
        current_size = 0

        # Sort parameters by size (largest first) for better bucketing
        sorted_params = sorted(params, key=lambda p: p.numel(), reverse=True)

        for param in sorted_params:
            param_size_mb = param.numel() * param.element_size() / (1024 * 1024)

            if current_size + param_size_mb > bucket_size_mb and current_bucket:
                buckets.append(current_bucket)
                current_bucket = [param]
                current_size = param_size_mb
            else:
                current_bucket.append(param)
                current_size += param_size_mb

        if current_bucket:
            buckets.append(current_bucket)

        return buckets

    # Create buckets for all parameters
    all_params = [p for p in model.parameters() if p.requires_grad]
    param_buckets = create_buckets(all_params)

    print0(f"Created {len(param_buckets)} gradient buckets")
    for i, bucket in enumerate(param_buckets):
        total_size = sum(p.numel() * p.element_size() for p in bucket) / (1024 * 1024)
        print0(f"Bucket {i}: {len(bucket)} params, {total_size:.1f} MB")

    # Bucket state tracking
    bucket_ready_count = [0] * len(param_buckets)
    bucket_handles = [None] * len(param_buckets)
    param_to_bucket = {}

    # Map each parameter to its bucket index
    for bucket_idx, bucket in enumerate(param_buckets):
        for param in bucket:
            param_to_bucket[param] = bucket_idx

    def _gradient_hook(param: Tensor):
        """Called when a parameter's gradient is ready"""
        if param.grad is None:
            return

        bucket_idx = param_to_bucket[param]
        bucket_ready_count[bucket_idx] += 1

        # Check if all parameters in this bucket are ready
        if bucket_ready_count[bucket_idx] == len(param_buckets[bucket_idx]):
            # All-reduce this bucket
            bucket_grads = [p.grad for p in param_buckets[bucket_idx]]

            # For multi-tensor operations, we can reduce them together
            if len(bucket_grads) == 1:
                handle = dist.all_reduce(bucket_grads[0], op=dist.ReduceOp.AVG, async_op=True)
            else:
                # Use multi-tensor all-reduce for efficiency
                handle = dist.all_reduce_coalesced(bucket_grads, op=dist.ReduceOp.AVG, async_op=True)

            bucket_handles[bucket_idx] = handle

    # Register hooks for all parameters
    print0("Registering bucketed gradient hooks...")
    for param in all_params:
        param.register_post_accumulate_grad_hook(_gradient_hook)

    def wait_for_gradients():
        """Wait for all gradient reductions to complete and reset bucket state"""
        for handle in bucket_handles:
            if handle is not None:
                handle.wait()

        # Reset state for next iteration
        for i in range(len(bucket_ready_count)):
            bucket_ready_count[i] = 0
            bucket_handles[i] = None

########################################
#            Warmup kernels            #
########################################

# Warmup the training kernels, then re-initialize the state so we aren't cheating
warmup_steps = 30
initial_state = dict(model=copy.deepcopy(model.state_dict()),
                     optimizers=[copy.deepcopy(opt.state_dict()) for opt in optimizers]) # save the initial state
train_loader = distributed_data_generator(args.train_files, world_size * args.train_seq_len, align_to_bos=True)
for _ in range(warmup_steps):
    inputs, targets = next(train_loader)
    warmup_loss = model(inputs, targets, get_window_size_blocks(1))
    warmup_loss.backward()
    for opt in optimizers:
        if isinstance(opt, DistLopt) and opt.lopt_type == "VeLO_CUDA":
            opt.step(warmup_loss)
        elif isinstance(opt, VeLO_CUDA):
            opt.step(warmup_loss)
        else:
            opt.step()
    model.zero_grad(set_to_none=True)
model.load_state_dict(initial_state["model"])
for opt, opt_state in zip(optimizers, initial_state["optimizers"]):
    opt.load_state_dict(opt_state)
del train_loader, initial_state

########################################
#        Training and validation       #
########################################

train_loader = distributed_data_generator(args.train_files, world_size * args.train_seq_len, align_to_bos=True)
training_time_ms = 0
# start the clock
torch.cuda.synchronize()
t0 = time.perf_counter()
# begin training
train_steps = args.num_iterations
for step in range(train_steps + 1):
    last_step = (step == train_steps)

    # --------------- VALIDATION SECTION -----------------
    if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.perf_counter() - t0)
        model.eval()
        val_batch_size = world_size * args.val_seq_len
        assert args.val_tokens % val_batch_size == 0
        val_steps = args.val_tokens // val_batch_size
        val_loader = distributed_data_generator(args.val_files, val_batch_size, align_to_bos=False)
        val_loss = 0
        with torch.no_grad():
            for _ in range(val_steps):
                inputs, targets = next(val_loader)
                val_loss += model(inputs, targets, get_window_size_blocks(step))
        val_loss /= val_steps
        del val_loader
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        print0(f"step:{step}/{train_steps} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step, 1):.2f}ms", console=True)
        model.train()
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.perf_counter()

    if last_step:
        if master_process and args.save_checkpoint:
            log = dict(step=step, code=code, model=model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
            os.makedirs(f"logs/{run_id}", exist_ok=True)
            torch.save(log, f"logs/{run_id}/state_step{step:06d}.pt")
        # the last step only has the validation loop, so break to avoid training
        break

    # --------------- TRAINING SECTION -----------------
    inputs, targets = next(train_loader)
    loss = model(inputs, targets, get_window_size_blocks(step))
    loss.backward()
    # set optimization hyperparameters
    scheduler_flags = [args.use_scheduler_opt1, args.use_scheduler_opt2] if len(optimizers) == 2 else [args.use_scheduler_opt1]
    for opt_idx, opt in enumerate(optimizers):
        if scheduler_flags[opt_idx]:
            for group in opt.param_groups:
                if opt_idx == 0 and args.use_cosine_lr:
                    group["lr"] = group["initial_lr"] * get_lr_cosine(step)
                else:
                    group["lr"] = group["initial_lr"] * get_lr(step)
    # momentum warmup for muon (only applies if using muon)
    if args.optimizer_mode == "muon":
        for group in optimizer2.param_groups:
            frac = min(step / 300, 1)
            group["momentum"] = (1 - frac) * 0.85 + frac * 0.95
    # step the optimizers
    for opt in optimizers:
        if isinstance(opt, DistLopt) and opt.lopt_type == "VeLO_CUDA":
            opt.step(loss)
        elif isinstance(opt, VeLO_CUDA):
            opt.step(loss)
        else:
            opt.step()
    # null the gradients
    model.zero_grad(set_to_none=True)
    # logging
    approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
    print0(f"step:{step+1}/{train_steps} train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms/(step + 1):.2f}ms", console=True)

print0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
       f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB", console=True)
dist.destroy_process_group()
