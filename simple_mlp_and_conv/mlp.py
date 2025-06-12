import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

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


class VarianceScaling:
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
            


class MLP(nn.Module):
    def __init__(
        self,
        in_channels,
        width=128,
        num_classes=10,
        nonlin=F.relu,
        bias=True,
    ):
        super(MLP, self).__init__()
        self.nonlin = nonlin
        self.fc_1 = nn.Linear(in_channels, width, bias=bias)
        self.fc_2 = nn.Linear(width, width, bias=bias)
        self.fc_3 = nn.Linear(width, width, bias=bias)
        self.fc_4 = nn.Linear(width, num_classes, bias=bias)  # Standard Linear instead of MuReadout
        self.reset_parameters()

    def reset_parameters(self):
        ini = VarianceScaling(scale=1.0, mode='fan_in', distribution='truncated_normal')
        ini.initialize(self.fc_1.weight)
        ini.initialize(self.fc_2.weight)
        ini.initialize(self.fc_3.weight)
        ini.initialize(self.fc_3.weight)
        # nn.init.ones_(self.fc_1.weight)
        # nn.init.ones_(self.fc_2.weight)
        # nn.init.ones_(self.fc_3.weight)
        # nn.init.ones_(self.fc_4.weight)
        nn.init.zeros_(self.fc_1.bias)
        nn.init.zeros_(self.fc_2.bias)
        nn.init.zeros_(self.fc_3.bias)
        nn.init.zeros_(self.fc_4.bias)

    def forward(self, x):
        out = self.fc_1(x.flatten(1))
        out = self.nonlin(out)
        pre_out = self.fc_2(out)
        out = self.nonlin(pre_out)
        pre_out = self.fc_3(out)
        out = self.nonlin(pre_out)
        pre_out = self.fc_4(out)
        out = pre_out
        return out
