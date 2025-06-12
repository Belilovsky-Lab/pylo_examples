import torch
import torch.nn as nn
import torch.nn.functional as F
from mup import MuReadout
from helpers import MupVarianceScaling

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
        use_mup=True,
    ):
        super(MuMLP, self).__init__()
        self.nonlin = nonlin
        self.input_mult = input_mult
        self.output_mult = output_mult
        self.init_std = init_std
        self.fc_1 = nn.Linear(in_channels, width, bias=bias)
        self.fc_2 = nn.Linear(width, width, bias=bias)
        self.fc_3 = nn.Linear(width, width, bias=bias)
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

    def forward(self, x):
        out = self.fc_1(x.flatten(1)) * self.input_mult
        out = self.nonlin(out)
        out = self.nonlin(self.fc_2(out))
        out = self.nonlin(self.fc_3(out))
        out = self.fc_4(out)
        return out