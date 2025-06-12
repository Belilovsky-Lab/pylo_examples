from functools import partial
from typing import Any, Callable, List, Optional, Type, Union
import math
import numpy as np

import torch
import torch.nn as nn
from torch import Tensor
from mup import MuReadout


def truncated_normal(lower, upper, shape=None, dtype=torch.float32, device=None):
    """
    Samples from a truncated normal distribution via inverse transform sampling.

    Args:
        lower (scalar or torch.Tensor): Lower bound(s) for truncation.
        upper (scalar or torch.Tensor): Upper bound(s) for truncation.
        shape (tuple or None): Desired output shape. If None, the broadcasted shape 
            of `lower` and `upper` is used.
        dtype (torch.dtype): The desired floating point type.
        device (torch.device or None): The device on which the tensor will be allocated.

    Returns:
        torch.Tensor: Samples from a truncated normal distribution with the given shape.
    """
    # Convert lower and upper to tensors (if they are not already)
    lower = torch.as_tensor(lower, dtype=dtype, device=device)
    upper = torch.as_tensor(upper, dtype=dtype, device=device)
    
    # If shape is not provided, use the broadcasted shape of lower and upper.
    if shape is None:
        shape = torch.broadcast_tensors(lower, upper)[0].shape
    else:
        # Optionally, you could add shape-checking logic here to ensure that
        # lower and upper are broadcastable to the provided shape.
        pass

    # Ensure that the dtype is a floating point type.
    if not torch.empty(0, dtype=dtype).is_floating_point():
        raise TypeError("truncated_normal only accepts floating point dtypes.")
    
    # Compute sqrt(2) as a tensor.
    sqrt2 = torch.tensor(np.sqrt(2), dtype=dtype, device=device)
    
    # Transform the truncation bounds using the error function.
    a = torch.erf(lower / sqrt2)
    b = torch.erf(upper / sqrt2)
    
    # Sample uniformly from the interval [a, b]. (The arithmetic here
    # broadcasts a and b to the desired shape.)
    u = a + (b - a) * torch.rand(shape, dtype=dtype, device=device)
    
    # Transform back using the inverse error function.
    out = sqrt2 * torch.erfinv(u)
    
    # To avoid any numerical issues, clamp the output so that it remains within
    # the open interval (lower, upper). Here we use torch.nextafter to compute the 
    # next representable floating point values:
    lower_bound = torch.nextafter(lower.detach(), torch.full_like(lower, float('inf')))
    upper_bound = torch.nextafter(upper.detach(), torch.full_like(upper, float('-inf')))
    out = torch.clamp(out, min=lower_bound, max=upper_bound)
    
    return out

class MupVarianceScaling:
    """Variance scaling initializer that adapts its scale to the shape of the initialized tensor.

    Args:
        scale (float): Scaling factor to multiply the variance by.
        mode (str): One of "fan_in", "fan_out", "fan_avg".
        distribution (str): One of "truncated_normal", "normal", or "uniform".
        fan_in_axes (Optional[Tuple[int]]): Optional sequence specifying the axes for fan-in calculation.
    """

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
        """Compute the fan-in and fan-out for the given shape."""
        dimensions = len(shape)
        if self.fan_in_axes is None:
            if dimensions == 2:  # For Linear layers
                fan_in, fan_out = shape[1], shape[0]
            else:  # For Conv layers
                receptive_field_size = math.prod(shape[2:])  # multiply all dimensions except first two
                fan_in = shape[1] * receptive_field_size
                fan_out = shape[0] * receptive_field_size
        else:
            # Custom fan-in based on specific axes
            fan_in = math.prod([shape[i] for i in self.fan_in_axes])
            fan_out = shape[0]
        return fan_in, fan_out

    def initialize(self, tensor):
        """Apply the initialization to the given tensor."""
        shape = tensor.shape
        fan_in, fan_out = self._compute_fans(shape)

        print("fan_in",fan_in)
        print("fan_out",fan_out)
        
        # Calculate the scale based on mode
        if self.mode == 'fan_in':
            scale = self.scale / max(1.0, fan_in)
        elif self.mode == 'fan_out':
            scale = self.scale / max(1.0, fan_out)
        else:  # fan_avg
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
        """Fill the tensor with values drawn from a truncated normal distribution."""
        with torch.no_grad():
            size = tensor.shape
            tensor.data.copy_(truncated_normal(lower=-2, upper=2, shape=size,))
            tensor.mul_(std).add_(mean)
            return tensor



__all__ = [
    "ResNet",
    "ResNet18_Weights",
    "ResNet34_Weights",
    "ResNet50_Weights",
    "ResNet101_Weights",
    "ResNet152_Weights",
    "ResNeXt50_32X4D_Weights",
    "ResNeXt101_32X8D_Weights",
    "ResNeXt101_64X4D_Weights",
    "Wide_ResNet50_2_Weights",
    "Wide_ResNet101_2_Weights",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "resnext101_64x4d",
    "wide_resnet50_2",
    "wide_resnet101_2",
]


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # if groups != 1 or base_width != 64:
        #     print(groups)
        #     print(base_width)
        #     raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        # if dilation > 1:
        #     raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 10,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        # _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.fc = MuReadout(512 * block.expansion, num_classes, 
                            output_mult=1.0, 
                            bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]


    def reset_parameters_all(self):
        ini = MupVarianceScaling(1.0, "fan_in", "truncated_normal")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                ini.initialize(m.weight,)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Initialize the final layer weights to zeros and bias to normal distribution
        nn.init.zeros_(self.fc.weight)
        nn.init.normal_(self.fc.bias, mean=0.0, std=1.0)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


# def _resnet(
#     block: Type[Union[BasicBlock, Bottleneck]],
#     layers: List[int],
#     weights: Optional[WeightsEnum],
#     progress: bool,
#     **kwargs: Any,
# ) -> ResNet:
#     if weights is not None:
#         _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

#     model = ResNet(block, layers, **kwargs)

#     if weights is not None:
#         model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

#     return model

def mu_resnet8(width: int = 1, **kwargs: Any) -> ResNet:
    """This is a wrapper for ResNet-18, which uses a width multiplier easily allow for scaling the network's width.

    Args:
        width: Width multiplier for the network. Default is 1.
    """
    return ResNet(BasicBlock, layers=[1, 1, 1, 1], width_per_group=width * 64, **kwargs)

def mu_resnet18(width: int = 1, **kwargs: Any) -> ResNet:
    """This is a wrapper for ResNet-18, which uses a width multiplier easily allow for scaling the network's width.

    Args:
        width: Width multiplier for the network. Default is 1.
    """
    return ResNet(BasicBlock, layers=[2, 2, 2, 2], width_per_group=width * 64, **kwargs)

def mu_resnet34(width: int = 1, **kwargs: Any) -> ResNet:
    """This is a wrapper for ResNet-34, which uses a width multiplier easily allow for scaling the network's width.

    Args:
        width: Width multiplier for the network. Default is 1.
    """
    return ResNet(BasicBlock, layers=[3, 4, 6, 3], width_per_group=width * 64, **kwargs)


def mu_resnet50(width: int = 1, **kwargs: Any) -> ResNet:
    """This is a wrapper for ResNet-50, which uses a width multiplier easily allow for scaling the network's width.

    Args:
        width: Width multiplier for the network. Default is 1.
    """
    return ResNet(Bottleneck, layers=[3, 4, 6, 3], width_per_group=width * 64, **kwargs)


def mu_resnet101(width: int = 1, **kwargs: Any) -> ResNet:
    """This is a wrapper for ResNet-101, which uses a width multiplier easily allow for scaling the network's width.

    Args:
        width: Width multiplier for the network. Default is 1.
    """
    return ResNet(Bottleneck, layers=[3, 4, 23, 3], width_per_group=width * 64, **kwargs)


def mu_resnet152(width: int = 1, **kwargs: Any) -> ResNet:
    """This is a wrapper for ResNet-152, which uses a width multiplier easily allow for scaling the network's width.

    Args:
        width: Width multiplier for the network. Default is 1.
    """
    return ResNet(Bottleneck, layers=[3, 8, 36, 3], width_per_group=width * 64, **kwargs)


def mu_resnext50_32x4d(width: int = 1, **kwargs: Any) -> ResNet:
    """This is a wrapper for ResNeXt-50 32x4d, which uses a width multiplier easily allow for scaling the network's width.

    Args:
        width: Width multiplier for the network. Default is 1.
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = width * 4
    return ResNet(Bottleneck, layers=[3, 4, 6, 3], **kwargs)


def mu_resnext101_32x8d(width: int = 1, **kwargs: Any) -> ResNet:
    """This is a wrapper for ResNeXt-101 32x8d, which uses a width multiplier easily allow for scaling the network's width.

    Args:
        width: Width multiplier for the network. Default is 1.
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = width * 8
    return ResNet(Bottleneck, layers=[3, 4, 23, 3], **kwargs)


def mu_resnext101_64x4d(width: int = 1, **kwargs: Any) -> ResNet:
    """This is a wrapper for ResNeXt-101 64x4d, which uses a width multiplier easily allow for scaling the network's width.

    Args:
        width: Width multiplier for the network. Default is 1.
    """
    kwargs["groups"] = 64
    kwargs["width_per_group"] = width * 4
    return ResNet(Bottleneck, layers=[3, 4, 23, 3], **kwargs)


def mu_wide_resnet50_2(width: int = 1, **kwargs: Any) -> ResNet:
    """This is a wrapper for Wide ResNet-50-2, which uses a width multiplier easily allow for scaling the network's width.

    Args:
        width: Width multiplier for the network. Default is 1.
    """
    kwargs["width_per_group"] = width * 64 * 2
    return ResNet(Bottleneck, layers=[3, 4, 6, 3], **kwargs)


def mu_wide_resnet101_2(width: int = 1, **kwargs: Any) -> ResNet:
    """This is a wrapper for Wide ResNet-101-2, which uses a width multiplier easily allow for scaling the network's width.

    Args:
        width: Width multiplier for the network. Default is 1.
    """
    kwargs["width_per_group"] = width * 64 * 2
    return ResNet(Bottleneck, layers=[3, 4, 23, 3], **kwargs)



# model = getattr(mugpt, config.model_name)(config, wm=config.MODEL.MuGPT2.width_mult)
# base_model = getattr(mugpt, config.model_name)(config, wm=config.MODEL.MuGPT2.base_width_mult)
# delta_model = getattr(mugpt, config.model_name)(config, wm=config.MODEL.MuGPT2.base_width_mult+1)
# # set the shapes to have base_width = 1 
# set_base_shapes(model, base_model, delta=delta_model,  override_base_dim=1)
# model.reset_parameters_all()

if __name__ == "__main__":
    from mup import get_shapes, make_base_shapes, set_base_shapes
    
    # Create models with different width multipliers
    model = mu_resnet50(width=1)  # Main model with width multiplier 2
    base_model = mu_resnet50(width=1)  # Base model with width multiplier 1
    delta_model = mu_resnet50(width=2)  # Delta model with width multiplier 2
    
    # Set the base shapes for μP
    set_base_shapes(model, base_model, delta=delta_model, override_base_dim=1)
    model.reset_parameters_all()
    
    # Print model information
    print(f"Main model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Base model parameters: {sum(p.numel() for p in base_model.parameters())}")
    print(f"Model architecture: {model}")

    for n,p in model.named_parameters():
        if p.infshape.main != None:
            with torch.no_grad():
                if 'conv' in n:
                    fan_in = np.prod(p.shape[1:])
                    print(n,  1/math.sqrt(fan_in), 1/fan_in, p.infshape.main.width_mult(), p.shape, p.infshape.ninf())
                    std, var = 1/math.sqrt(fan_in), 1/fan_in
                    print("mean {:4f} std {:4f}=={:4f} var {:4f}=={:4f}".format(p.mean(),p.std(),std,p.var(),var))







# _COMMON_META = {
#     "min_size": (1, 1),
#     "categories": _IMAGENET_CATEGORIES,
# }


# class ResNet18_Weights(WeightsEnum):
#     IMAGENET1K_V1 = Weights(
#         url="https://download.pytorch.org/models/resnet18-f37072fd.pth",
#         transforms=partial(ImageClassification, crop_size=224),
#         meta={
#             **_COMMON_META,
#             "num_params": 11689512,
#             "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnet",
#             "_metrics": {
#                 "ImageNet-1K": {
#                     "acc@1": 69.758,
#                     "acc@5": 89.078,
#                 }
#             },
#             "_ops": 1.814,
#             "_file_size": 44.661,
#             "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
#         },
#     )
#     DEFAULT = IMAGENET1K_V1


# class ResNet34_Weights(WeightsEnum):
#     IMAGENET1K_V1 = Weights(
#         url="https://download.pytorch.org/models/resnet34-b627a593.pth",
#         transforms=partial(ImageClassification, crop_size=224),
#         meta={
#             **_COMMON_META,
#             "num_params": 21797672,
#             "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnet",
#             "_metrics": {
#                 "ImageNet-1K": {
#                     "acc@1": 73.314,
#                     "acc@5": 91.420,
#                 }
#             },
#             "_ops": 3.664,
#             "_file_size": 83.275,
#             "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
#         },
#     )
#     DEFAULT = IMAGENET1K_V1


# class ResNet50_Weights(WeightsEnum):
#     IMAGENET1K_V1 = Weights(
#         url="https://download.pytorch.org/models/resnet50-0676ba61.pth",
#         transforms=partial(ImageClassification, crop_size=224),
#         meta={
#             **_COMMON_META,
#             "num_params": 25557032,
#             "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnet",
#             "_metrics": {
#                 "ImageNet-1K": {
#                     "acc@1": 76.130,
#                     "acc@5": 92.862,
#                 }
#             },
#             "_ops": 4.089,
#             "_file_size": 97.781,
#             "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
#         },
#     )
#     IMAGENET1K_V2 = Weights(
#         url="https://download.pytorch.org/models/resnet50-11ad3fa6.pth",
#         transforms=partial(ImageClassification, crop_size=224, resize_size=232),
#         meta={
#             **_COMMON_META,
#             "num_params": 25557032,
#             "recipe": "https://github.com/pytorch/vision/issues/3995#issuecomment-1013906621",
#             "_metrics": {
#                 "ImageNet-1K": {
#                     "acc@1": 80.858,
#                     "acc@5": 95.434,
#                 }
#             },
#             "_ops": 4.089,
#             "_file_size": 97.79,
#             "_docs": """
#                 These weights improve upon the results of the original paper by using TorchVision's `new training recipe
#                 <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
#             """,
#         },
#     )
#     DEFAULT = IMAGENET1K_V2


# class ResNet101_Weights(WeightsEnum):
#     IMAGENET1K_V1 = Weights(
#         url="https://download.pytorch.org/models/resnet101-63fe2227.pth",
#         transforms=partial(ImageClassification, crop_size=224),
#         meta={
#             **_COMMON_META,
#             "num_params": 44549160,
#             "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnet",
#             "_metrics": {
#                 "ImageNet-1K": {
#                     "acc@1": 77.374,
#                     "acc@5": 93.546,
#                 }
#             },
#             "_ops": 7.801,
#             "_file_size": 170.511,
#             "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
#         },
#     )
#     IMAGENET1K_V2 = Weights(
#         url="https://download.pytorch.org/models/resnet101-cd907fc2.pth",
#         transforms=partial(ImageClassification, crop_size=224, resize_size=232),
#         meta={
#             **_COMMON_META,
#             "num_params": 44549160,
#             "recipe": "https://github.com/pytorch/vision/issues/3995#new-recipe",
#             "_metrics": {
#                 "ImageNet-1K": {
#                     "acc@1": 81.886,
#                     "acc@5": 95.780,
#                 }
#             },
#             "_ops": 7.801,
#             "_file_size": 170.53,
#             "_docs": """
#                 These weights improve upon the results of the original paper by using TorchVision's `new training recipe
#                 <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
#             """,
#         },
#     )
#     DEFAULT = IMAGENET1K_V2


# class ResNet152_Weights(WeightsEnum):
#     IMAGENET1K_V1 = Weights(
#         url="https://download.pytorch.org/models/resnet152-394f9c45.pth",
#         transforms=partial(ImageClassification, crop_size=224),
#         meta={
#             **_COMMON_META,
#             "num_params": 60192808,
#             "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnet",
#             "_metrics": {
#                 "ImageNet-1K": {
#                     "acc@1": 78.312,
#                     "acc@5": 94.046,
#                 }
#             },
#             "_ops": 11.514,
#             "_file_size": 230.434,
#             "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
#         },
#     )
#     IMAGENET1K_V2 = Weights(
#         url="https://download.pytorch.org/models/resnet152-f82ba261.pth",
#         transforms=partial(ImageClassification, crop_size=224, resize_size=232),
#         meta={
#             **_COMMON_META,
#             "num_params": 60192808,
#             "recipe": "https://github.com/pytorch/vision/issues/3995#new-recipe",
#             "_metrics": {
#                 "ImageNet-1K": {
#                     "acc@1": 82.284,
#                     "acc@5": 96.002,
#                 }
#             },
#             "_ops": 11.514,
#             "_file_size": 230.474,
#             "_docs": """
#                 These weights improve upon the results of the original paper by using TorchVision's `new training recipe
#                 <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
#             """,
#         },
#     )
#     DEFAULT = IMAGENET1K_V2


# class ResNeXt50_32X4D_Weights(WeightsEnum):
#     IMAGENET1K_V1 = Weights(
#         url="https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
#         transforms=partial(ImageClassification, crop_size=224),
#         meta={
#             **_COMMON_META,
#             "num_params": 25028904,
#             "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnext",
#             "_metrics": {
#                 "ImageNet-1K": {
#                     "acc@1": 77.618,
#                     "acc@5": 93.698,
#                 }
#             },
#             "_ops": 4.23,
#             "_file_size": 95.789,
#             "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
#         },
#     )
#     IMAGENET1K_V2 = Weights(
#         url="https://download.pytorch.org/models/resnext50_32x4d-1a0047aa.pth",
#         transforms=partial(ImageClassification, crop_size=224, resize_size=232),
#         meta={
#             **_COMMON_META,
#             "num_params": 25028904,
#             "recipe": "https://github.com/pytorch/vision/issues/3995#new-recipe",
#             "_metrics": {
#                 "ImageNet-1K": {
#                     "acc@1": 81.198,
#                     "acc@5": 95.340,
#                 }
#             },
#             "_ops": 4.23,
#             "_file_size": 95.833,
#             "_docs": """
#                 These weights improve upon the results of the original paper by using TorchVision's `new training recipe
#                 <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
#             """,
#         },
#     )
#     DEFAULT = IMAGENET1K_V2


# class ResNeXt101_32X8D_Weights(WeightsEnum):
#     IMAGENET1K_V1 = Weights(
#         url="https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
#         transforms=partial(ImageClassification, crop_size=224),
#         meta={
#             **_COMMON_META,
#             "num_params": 88791336,
#             "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnext",
#             "_metrics": {
#                 "ImageNet-1K": {
#                     "acc@1": 79.312,
#                     "acc@5": 94.526,
#                 }
#             },
#             "_ops": 16.414,
#             "_file_size": 339.586,
#             "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
#         },
#     )
#     IMAGENET1K_V2 = Weights(
#         url="https://download.pytorch.org/models/resnext101_32x8d-110c445d.pth",
#         transforms=partial(ImageClassification, crop_size=224, resize_size=232),
#         meta={
#             **_COMMON_META,
#             "num_params": 88791336,
#             "recipe": "https://github.com/pytorch/vision/issues/3995#new-recipe-with-fixres",
#             "_metrics": {
#                 "ImageNet-1K": {
#                     "acc@1": 82.834,
#                     "acc@5": 96.228,
#                 }
#             },
#             "_ops": 16.414,
#             "_file_size": 339.673,
#             "_docs": """
#                 These weights improve upon the results of the original paper by using TorchVision's `new training recipe
#                 <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
#             """,
#         },
#     )
#     DEFAULT = IMAGENET1K_V2


# class ResNeXt101_64X4D_Weights(WeightsEnum):
#     IMAGENET1K_V1 = Weights(
#         url="https://download.pytorch.org/models/resnext101_64x4d-173b62eb.pth",
#         transforms=partial(ImageClassification, crop_size=224, resize_size=232),
#         meta={
#             **_COMMON_META,
#             "num_params": 83455272,
#             "recipe": "https://github.com/pytorch/vision/pull/5935",
#             "_metrics": {
#                 "ImageNet-1K": {
#                     "acc@1": 83.246,
#                     "acc@5": 96.454,
#                 }
#             },
#             "_ops": 15.46,
#             "_file_size": 319.318,
#             "_docs": """
#                 These weights were trained from scratch by using TorchVision's `new training recipe
#                 <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
#             """,
#         },
#     )
#     DEFAULT = IMAGENET1K_V1


# class Wide_ResNet50_2_Weights(WeightsEnum):
#     IMAGENET1K_V1 = Weights(
#         url="https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
#         transforms=partial(ImageClassification, crop_size=224),
#         meta={
#             **_COMMON_META,
#             "num_params": 68883240,
#             "recipe": "https://github.com/pytorch/vision/pull/912#issue-445437439",
#             "_metrics": {
#                 "ImageNet-1K": {
#                     "acc@1": 78.468,
#                     "acc@5": 94.086,
#                 }
#             },
#             "_ops": 11.398,
#             "_file_size": 131.82,
#             "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
#         },
#     )
#     IMAGENET1K_V2 = Weights(
#         url="https://download.pytorch.org/models/wide_resnet50_2-9ba9bcbe.pth",
#         transforms=partial(ImageClassification, crop_size=224, resize_size=232),
#         meta={
#             **_COMMON_META,
#             "num_params": 68883240,
#             "recipe": "https://github.com/pytorch/vision/issues/3995#new-recipe-with-fixres",
#             "_metrics": {
#                 "ImageNet-1K": {
#                     "acc@1": 81.602,
#                     "acc@5": 95.758,
#                 }
#             },
#             "_ops": 11.398,
#             "_file_size": 263.124,
#             "_docs": """
#                 These weights improve upon the results of the original paper by using TorchVision's `new training recipe
#                 <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
#             """,
#         },
#     )
#     DEFAULT = IMAGENET1K_V2


# class Wide_ResNet101_2_Weights(WeightsEnum):
#     IMAGENET1K_V1 = Weights(
#         url="https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
#         transforms=partial(ImageClassification, crop_size=224),
#         meta={
#             **_COMMON_META,
#             "num_params": 126886696,
#             "recipe": "https://github.com/pytorch/vision/pull/912#issue-445437439",
#             "_metrics": {
#                 "ImageNet-1K": {
#                     "acc@1": 78.848,
#                     "acc@5": 94.284,
#                 }
#             },
#             "_ops": 22.753,
#             "_file_size": 242.896,
#             "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
#         },
#     )
#     IMAGENET1K_V2 = Weights(
#         url="https://download.pytorch.org/models/wide_resnet101_2-d733dc28.pth",
#         transforms=partial(ImageClassification, crop_size=224, resize_size=232),
#         meta={
#             **_COMMON_META,
#             "num_params": 126886696,
#             "recipe": "https://github.com/pytorch/vision/issues/3995#new-recipe",
#             "_metrics": {
#                 "ImageNet-1K": {
#                     "acc@1": 82.510,
#                     "acc@5": 96.020,
#                 }
#             },
#             "_ops": 22.753,
#             "_file_size": 484.747,
#             "_docs": """
#                 These weights improve upon the results of the original paper by using TorchVision's `new training recipe
#                 <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
#             """,
#         },
#     )
#     DEFAULT = IMAGENET1K_V2

