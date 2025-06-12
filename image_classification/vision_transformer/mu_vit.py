import math
import numpy as np
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, NamedTuple, Optional

import torch
import torch.nn as nn
from torch import Tensor
from mup.layer import MuReadout

from torchvision.ops.misc import Conv2dNormActivation, MLP
from torchvision.transforms._presets import ImageClassification, InterpolationMode
from torchvision.utils import _log_api_usage_once
from torchvision.models._api import register_model, Weights, WeightsEnum
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models._utils import _ovewrite_named_param, handle_legacy_interface


__all__ = [
    "VisionTransformer",
    "ViT_B_16_Weights",
    "ViT_B_32_Weights",
    "ViT_L_16_Weights",
    "ViT_L_32_Weights",
    "ViT_H_14_Weights",
    "vit_b_16",
    "vit_b_32",
    "vit_l_16",
    "vit_l_32",
    "vit_h_14",
]


class ConvStemConfig(NamedTuple):
    out_channels: int
    kernel_size: int
    stride: int
    norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d
    activation_layer: Callable[..., nn.Module] = nn.ReLU


class MLPBlock(MLP):
    """Transformer MLP block."""

    _version = 2

    def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
        super().__init__(in_dim, [mlp_dim, in_dim], activation_layer=nn.GELU, inplace=None, dropout=dropout)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

def truncated_normal(lower, upper, shape=None, dtype=torch.float32, device=None):
    lower = torch.as_tensor(lower, dtype=dtype, device=device)
    upper = torch.as_tensor(upper, dtype=dtype, device=device)
    if shape is None:
        shape = torch.broadcast_tensors(lower, upper)[0].shape
    if not torch.empty(0, dtype=dtype).is_floating_point():
        raise TypeError("truncated_normal only accepts floating point dtypes.")
    sqrt2 = torch.tensor(np.sqrt(2), dtype=dtype, device=device)
    a = torch.erf(lower / sqrt2)
    b = torch.erf(upper / sqrt2)
    u = a + (b - a) * torch.rand(shape, dtype=dtype, device=device)
    out = sqrt2 * torch.erfinv(u)
    lower_bound = torch.nextafter(lower.detach(), torch.full_like(lower, float('inf')))
    upper_bound = torch.nextafter(upper.detach(), torch.full_like(upper, float('-inf')))
    out = torch.clamp(out, min=lower_bound, max=upper_bound)
    return out


class MupVarianceScaling:
    def __init__(self, scale=1.0, mode='fan_in', distribution='truncated_normal', fan_in_axes=None):
        if scale < 0.0:
            raise ValueError('`scale` must be a positive float.')
        if mode not in {'fan_in', 'fan_out', 'fan_avg'}:
            raise ValueError(f'Invalid `mode` argument: {mode}')
        if distribution not in {'normal', 'truncated_normal', 'uniform'}:
            raise ValueError(f'Invalid `distribution` argument: {distribution}')
        self.scale = scale
        self.mode = mode
        self.distribution = distribution
        self.fan_in_axes = fan_in_axes
    def _compute_fans(self, shape):
        dimensions = len(shape)
        if self.fan_in_axes is None:
            if dimensions == 2:
                fan_in, fan_out = shape[1], shape[0]
            else:
                receptive_field_size = math.prod(shape[2:])
                fan_in = shape[1] * receptive_field_size
                fan_out = shape[0] * receptive_field_size
        else:
            fan_in = math.prod([shape[i] for i in self.fan_in_axes])
            fan_out = shape[0]
        return fan_in, fan_out
    def initialize(self, tensor):
        shape = tensor.shape
        fan_in, fan_out = self._compute_fans(shape)
        if self.mode == 'fan_in':
            scale = self.scale / max(1.0, fan_in)
        elif self.mode == 'fan_out':
            scale = self.scale / max(1.0, fan_out)
        else:
            scale = self.scale / max(1.0, (fan_in + fan_out) / 2.0)
        if self.distribution == 'truncated_normal':
            stddev = math.sqrt(scale)
            return self._truncated_normal_(tensor, mean=0.0, std=stddev)
        elif self.distribution == 'normal':
            stddev = math.sqrt(scale)
            return torch.nn.init.normal_(tensor, mean=0.0, std=stddev)
        elif self.distribution == 'uniform':
            limit = math.sqrt(3.0 * scale)
            return torch.nn.init.uniform_(tensor, a=-limit, b=limit)
    @staticmethod
    def _truncated_normal_(tensor, mean=0.0, std=1.0):
        with torch.no_grad():
            size = tensor.shape
            tensor.data.copy_(truncated_normal(lower=-2, upper=2, shape=size,))
            tensor.mul_(std).add_(mean)
            return tensor


class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x, _ = self.self_attention(x, x, x, need_weights=False)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y


class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # from BERT
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        input = input + self.pos_embedding
        return self.ln(self.layers(self.dropout(input)))


class MuVisionTransformer(nn.Module):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929."""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_classes: int = 1000,
        representation_size: Optional[int] = None,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        conv_stem_configs: Optional[list[ConvStemConfig]] = None,
    ):
        super().__init__()
        _log_api_usage_once(self)
        torch._assert(image_size % patch_size == 0, "Input shape indivisible by patch size!")
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.num_classes = num_classes
        self.representation_size = representation_size
        self.norm_layer = norm_layer

        if conv_stem_configs is not None:
            # As per https://arxiv.org/abs/2106.14881
            seq_proj = nn.Sequential()
            prev_channels = 3
            for i, conv_stem_layer_config in enumerate(conv_stem_configs):
                seq_proj.add_module(
                    f"conv_bn_relu_{i}",
                    Conv2dNormActivation(
                        in_channels=prev_channels,
                        out_channels=conv_stem_layer_config.out_channels,
                        kernel_size=conv_stem_layer_config.kernel_size,
                        stride=conv_stem_layer_config.stride,
                        norm_layer=conv_stem_layer_config.norm_layer,
                        activation_layer=conv_stem_layer_config.activation_layer,
                    ),
                )
                prev_channels = conv_stem_layer_config.out_channels
            seq_proj.add_module(
                "conv_last", nn.Conv2d(in_channels=prev_channels, out_channels=hidden_dim, kernel_size=1)
            )
            self.conv_proj: nn.Module = seq_proj
        else:
            self.conv_proj = nn.Conv2d(
                in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size
            )

        seq_length = (image_size // patch_size) ** 2

        # Add a class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        seq_length += 1

        self.encoder = Encoder(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer,
        )
        self.seq_length = seq_length

        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        if representation_size is None:
            heads_layers["head"] = MuReadout(hidden_dim, num_classes, output_mult=1.0, bias=True)
        else:
            heads_layers["pre_logits"] = nn.Linear(hidden_dim, representation_size)
            heads_layers["act"] = nn.Tanh()
            heads_layers["head"] = MuReadout(representation_size, num_classes, output_mult=1.0, bias=True)
        self.heads = nn.Sequential(heads_layers)

        if isinstance(self.conv_proj, nn.Conv2d):
            # Init the patchify stem
            fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
            nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
            if self.conv_proj.bias is not None:
                nn.init.zeros_(self.conv_proj.bias)
        elif self.conv_proj.conv_last is not None and isinstance(self.conv_proj.conv_last, nn.Conv2d):
            # Init the last 1x1 conv of the conv stem
            nn.init.normal_(
                self.conv_proj.conv_last.weight, mean=0.0, std=math.sqrt(2.0 / self.conv_proj.conv_last.out_channels)
            )
            if self.conv_proj.conv_last.bias is not None:
                nn.init.zeros_(self.conv_proj.conv_last.bias)

        if hasattr(self.heads, "pre_logits") and isinstance(self.heads.pre_logits, nn.Linear):
            fan_in = self.heads.pre_logits.in_features
            nn.init.trunc_normal_(self.heads.pre_logits.weight, std=math.sqrt(1 / fan_in))
            nn.init.zeros_(self.heads.pre_logits.bias)

        if isinstance(self.heads.head, MuReadout):
            nn.init.zeros_(self.heads.head.weight)
            if self.heads.head.bias is not None:
                nn.init.normal_(self.heads.head.bias, mean=0.0, std=1.0)

    def reset_parameters_all(self):
        ini = MupVarianceScaling(1.0, "fan_in", "truncated_normal")
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.modules.linear.NonDynamicallyQuantizableLinear):
                ini.initialize(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, mean=0.0, std=1.0)
            elif isinstance(m, nn.Conv2d):
                ini.initialize(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, mean=0.0, std=1.0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.MultiheadAttention):
                # Initialize the output projection
                if hasattr(m, 'out_proj'):
                    ini.initialize(m.out_proj.weight)
                    if m.out_proj.bias is not None:
                        nn.init.normal_(m.out_proj.bias, mean=0.0, std=1.0)
                
                # For completeness, check for other attributes that might exist
                # in different PyTorch versions or configurations
                if hasattr(m, 'in_proj_weight'):
                    ini.initialize(m.in_proj_weight)
                # Check if attributes exist before trying to initialize them
                if hasattr(m, 'q_proj_weight') and m.q_proj_weight is not None:
                    ini.initialize(m.q_proj_weight)
                if hasattr(m, 'k_proj_weight') and m.k_proj_weight is not None:
                    ini.initialize(m.k_proj_weight)
                if hasattr(m, 'v_proj_weight') and m.v_proj_weight is not None:
                    ini.initialize(m.v_proj_weight)
                if hasattr(m, 'in_proj_bias') and m.in_proj_bias is not None:
                    nn.init.normal_(m.in_proj_bias, mean=0.0, std=1.0)
                if hasattr(m, 'bias_k') and m.bias_k is not None:
                    nn.init.normal_(m.bias_k, mean=0.0, std=1.0)
                if hasattr(m, 'bias_v') and m.bias_v is not None:
                    nn.init.normal_(m.bias_v, mean=0.0, std=1.0)
            elif isinstance(m, MLPBlock):
                # MLPBlock is already handled by its constituent Linear layers
                pass
            elif isinstance(m, nn.GELU):
                # GELU has no parameters to initialize
                pass
            elif isinstance(m, nn.Dropout):
                # Dropout has no parameters to initialize
                pass
            elif isinstance(m, (nn.Sequential, EncoderBlock, Encoder, MuVisionTransformer)):
                # Container modules are handled by iterating through their children
                pass
                
        # Output layer: zero weights, bias normal
        if hasattr(self.heads, "head") and isinstance(self.heads.head, MuReadout):
            nn.init.zeros_(self.heads.head.weight)
            if self.heads.head.bias is not None:
                nn.init.normal_(self.heads.head.bias, mean=0.0, std=1.0)

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x

    def forward(self, x: torch.Tensor):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        x = self.heads(x)

        return x


def _vision_transformer(
    patch_size: int,
    num_layers: int,
    num_heads: int,
    hidden_dim: int,
    mlp_dim: int,
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> MuVisionTransformer:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
        assert weights.meta["min_size"][0] == weights.meta["min_size"][1]
        _ovewrite_named_param(kwargs, "image_size", weights.meta["min_size"][0])
    image_size = kwargs.pop("image_size", 224)

    model = MuVisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        mlp_dim=mlp_dim,
        **kwargs,
    )

    if weights:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

    return model




def interpolate_embeddings(
    image_size: int,
    patch_size: int,
    model_state: "OrderedDict[str, torch.Tensor]",
    interpolation_mode: str = "bicubic",
    reset_heads: bool = False,
) -> "OrderedDict[str, torch.Tensor]":
    """This function helps interpolate positional embeddings during checkpoint loading,
    especially when you want to apply a pre-trained model on images with different resolution.

    Args:
        image_size (int): Image size of the new model.
        patch_size (int): Patch size of the new model.
        model_state (OrderedDict[str, torch.Tensor]): State dict of the pre-trained model.
        interpolation_mode (str): The algorithm used for upsampling. Default: bicubic.
        reset_heads (bool): If true, not copying the state of heads. Default: False.

    Returns:
        OrderedDict[str, torch.Tensor]: A state dict which can be loaded into the new model.
    """
    # Shape of pos_embedding is (1, seq_length, hidden_dim)
    pos_embedding = model_state["encoder.pos_embedding"]
    n, seq_length, hidden_dim = pos_embedding.shape
    if n != 1:
        raise ValueError(f"Unexpected position embedding shape: {pos_embedding.shape}")

    new_seq_length = (image_size // patch_size) ** 2 + 1

    # Need to interpolate the weights for the position embedding.
    # We do this by reshaping the positions embeddings to a 2d grid, performing
    # an interpolation in the (h, w) space and then reshaping back to a 1d grid.
    if new_seq_length != seq_length:
        # The class token embedding shouldn't be interpolated, so we split it up.
        seq_length -= 1
        new_seq_length -= 1
        pos_embedding_token = pos_embedding[:, :1, :]
        pos_embedding_img = pos_embedding[:, 1:, :]

        # (1, seq_length, hidden_dim) -> (1, hidden_dim, seq_length)
        pos_embedding_img = pos_embedding_img.permute(0, 2, 1)
        seq_length_1d = int(math.sqrt(seq_length))
        if seq_length_1d * seq_length_1d != seq_length:
            raise ValueError(
                f"seq_length is not a perfect square! Instead got seq_length_1d * seq_length_1d = {seq_length_1d * seq_length_1d } and seq_length = {seq_length}"
            )

        # (1, hidden_dim, seq_length) -> (1, hidden_dim, seq_l_1d, seq_l_1d)
        pos_embedding_img = pos_embedding_img.reshape(1, hidden_dim, seq_length_1d, seq_length_1d)
        new_seq_length_1d = image_size // patch_size

        # Perform interpolation.
        # (1, hidden_dim, seq_l_1d, seq_l_1d) -> (1, hidden_dim, new_seq_l_1d, new_seq_l_1d)
        new_pos_embedding_img = nn.functional.interpolate(
            pos_embedding_img,
            size=new_seq_length_1d,
            mode=interpolation_mode,
            align_corners=True,
        )

        # (1, hidden_dim, new_seq_l_1d, new_seq_l_1d) -> (1, hidden_dim, new_seq_length)
        new_pos_embedding_img = new_pos_embedding_img.reshape(1, hidden_dim, new_seq_length)

        # (1, hidden_dim, new_seq_length) -> (1, new_seq_length, hidden_dim)
        new_pos_embedding_img = new_pos_embedding_img.permute(0, 2, 1)
        new_pos_embedding = torch.cat([pos_embedding_token, new_pos_embedding_img], dim=1)

        model_state["encoder.pos_embedding"] = new_pos_embedding

        if reset_heads:
            model_state_copy: "OrderedDict[str, torch.Tensor]" = OrderedDict()
            for k, v in model_state.items():
                if not k.startswith("heads"):
                    model_state_copy[k] = v
            model_state = model_state_copy

    return model_state


def mu_vit(image_size=224, patch_size=16, num_layers=12, num_heads=12, hidden_dim=768, mlp_dim=3072, **kwargs):
    return MuVisionTransformer(
        image_size=image_size, patch_size=patch_size, num_layers=num_layers, 
        num_heads=num_heads, hidden_dim=hidden_dim, mlp_dim=mlp_dim, **kwargs
    )

def mu_vit_b_16(**kwargs):
    return MuVisionTransformer(
        image_size=224, patch_size=16, num_layers=12, num_heads=12, hidden_dim=768, mlp_dim=3072, **kwargs
    )


def mu_vit_b_32(**kwargs):
    return MuVisionTransformer(
        image_size=224, patch_size=32, num_layers=12, num_heads=12, hidden_dim=768, mlp_dim=3072, **kwargs
    )


def mu_vit_l_16(**kwargs):
    return MuVisionTransformer(
        image_size=224, patch_size=16, num_layers=24, num_heads=16, hidden_dim=1024, mlp_dim=4096, **kwargs
    )


def mu_vit_l_32(**kwargs):
    return MuVisionTransformer(
        image_size=224, patch_size=32, num_layers=24, num_heads=16, hidden_dim=1024, mlp_dim=4096, **kwargs
    )


def mu_vit_h_14(**kwargs):
    return MuVisionTransformer(
        image_size=518, patch_size=14, num_layers=32, num_heads=16, hidden_dim=1280, mlp_dim=5120, **kwargs
    )


if __name__ == "__main__":
    from mup import get_shapes, make_base_shapes, set_base_shapes

    # Create models with different width multipliers
    model = mu_vit(hidden_dim=768,mlp_dim=3072,)  # Main model with width multiplier 2

    base_model = mu_vit(hidden_dim=24, mlp_dim=24*4,)  # Base model with width multiplier 1
    delta_model = mu_vit(hidden_dim=36, mlp_dim=36*4,) # Delta model with width multiplier 2
    
    
    # Print model information
    print(f"Main model parameters: {sum(p.numel() for p in model.parameters())/1e6}")
    print(f"Base model parameters: {sum(p.numel() for p in base_model.parameters())}")
    print(f"Model architecture: {model}")

    # Set the base shapes for μP
    set_base_shapes(model, base_model, delta=delta_model, override_base_dim=1)
    model.reset_parameters_all()
    tps = []
    # Print the class type of the specified layer
    for name, module in model.named_modules():
        print(name, type(module))
        tps.append(type(module))
        # if name == "encoder.layers.encoder_layer_6.self_attention":
        #     print(f"Type of encoder.layers.encoder_layer_6.self_attention: {type(module)}")
        #     # Print additional information about the module
        #     print(f"Module structure: {module}")

        #     import pdb; pdb.set_trace()
        #     if hasattr(module, 'in_proj_weight'):
        #         print(f"in_proj_weight shape: {module.in_proj_weight.shape}")
        #     break
        # else:
        #     print (name, "Layer 'encoder.layers.encoder_layer_6.self_attention' not found in model modules")
    import pprint
    pprint.pprint(set(tps))
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6}")
    for n, p in model.named_parameters():
        # print(f"{n}: mean={p.mean():.4f}, std={p.std():.4f}, shape={p.shape}")
        if 'weight' in n and len(p.shape) > 1:
            fan_in = np.prod(p.shape[1:])
            # print(n,  1/math.sqrt(fan_in), 1/fan_in, p.infshape.main.width_mult(), p.shape, p.infshape.ninf())
            std, var = 1/math.sqrt(fan_in), 1/fan_in
            diff = (p.std() - std).abs()
            if diff > 1e-4:
                print("mean {:4f} std {:4f}=={:4f} var {:4f}=={:4f} || {} {} {} {}".format( p.mean(),p.std(),std,p.var(),var,p.shape, p.infshape.ninf(), n, type(p)))
        # elif 'bias' in n:
        #     print("mean {:4f} std {:4f}=={:4f} var {:4f}=={:4f} || {} {} {} {}".format( p.mean(),p.std(),std,p.var(),var,p.shape, p.infshape.ninf(), n, type(p)))

