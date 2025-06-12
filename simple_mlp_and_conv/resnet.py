
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BasicBlock(nn.Module):
    """
    Basic Block for ResNet18 and ResNet34
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
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
    """
    Bottleneck Block for ResNet50, ResNet101, and ResNet152
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        width = int(planes * (base_width / 64.)) * groups
        
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

    def forward(self, x):
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

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class ResNet(nn.Module):
    """
    ResNet implementation that supports ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
    """
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        
        if replace_stride_with_dilation is None:
            # Each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        
        self.groups = groups
        self.base_width = width_per_group
        
        # Initial convolution and pooling
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        
        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
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
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Classification head
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

    def get_feature_maps(self, x):
        """
        Get intermediate feature maps for visualization or feature extraction
        """
        features = {}
        
        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features['conv1'] = x
        x = self.maxpool(x)
        features['maxpool'] = x

        # Residual layers
        x = self.layer1(x)
        features['layer1'] = x
        x = self.layer2(x)
        features['layer2'] = x
        x = self.layer3(x)
        features['layer3'] = x
        x = self.layer4(x)
        features['layer4'] = x

        # Global average pooling
        x = self.avgpool(x)
        features['avgpool'] = x
        
        return features

def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model

def resnet18(pretrained=False, progress=True, **kwargs):
    """ResNet-18 model from
    "Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)

def resnet34(pretrained=False, progress=True, **kwargs):
    """ResNet-34 model from
    "Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)

def resnet50(pretrained=False, progress=True, **kwargs):
    """ResNet-50 model from
    "Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)

def resnet101(pretrained=False, progress=True, **kwargs):
    """ResNet-101 model from
    "Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)

def resnet152(pretrained=False, progress=True, **kwargs):
    """ResNet-152 model from
    "Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress, **kwargs)

def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    """ResNeXt-50 32x4d model from
    "Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)

def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    """Wide ResNet-50-2 model from
    "Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)

class ResNet50(nn.Module):
    """
    Simplified ResNet50 class for easy instantiation
    """
    def __init__(self, num_classes=1000, zero_init_residual=False):
        super(ResNet50, self).__init__()
        self.resnet = resnet50(num_classes=num_classes, zero_init_residual=zero_init_residual)
    
    def forward(self, x):
        return self.resnet(x)
    
    def get_feature_maps(self, x):
        return self.resnet.get_feature_maps(x)

# Utility functions for model analysis
def count_parameters(model):
    """Count the number of trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def model_summary(model, input_size=(3, 224, 224)):
    """Print a summary of the model architecture"""
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)
            
            m_key = f"{class_name}-{module_idx+1}"
            summary[m_key] = {}
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = -1
            
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = -1

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    # Create a summary dict
    summary = {}
    hooks = []

    # Register hooks
    model.apply(register_hook)

    # Make a forward pass
    model.eval()
    with torch.no_grad():
        x = torch.randn(1, *input_size)
        model(x)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Print summary
    print("=" * 70)
    print(f"{'Layer (type)':<25} {'Output Shape':<25} {'Param #':<15}")
    print("=" * 70)
    
    total_params = 0
    total_output = 0
    trainable_params = 0
    
    for layer in summary:
        output_shape = str(summary[layer]["output_shape"])
        num_params = summary[layer]["nb_params"]
        
        total_params += num_params
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"]:
                trainable_params += num_params
        
        print(f"{layer:<25} {output_shape:<25} {num_params:<15,}")

    print("=" * 70)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Non-trainable params: {total_params - trainable_params:,}")
    print("=" * 70)

# Example usage and testing
def test_resnet50():
    """Test the ResNet50 implementation"""
    print("Testing ResNet50 implementation...")
    
    # Create model
    model = ResNet50(num_classes=1000)
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    
    print(f"Input shape: {x.shape}")
    
    with torch.no_grad():
        output = model(x)
        print(f"Output shape: {output.shape}")
        
        # Test feature extraction
        features = model.get_feature_maps(x)
        print("\nFeature map shapes:")
        for name, feat in features.items():
            print(f"  {name}: {feat.shape}")
    
    # Count parameters
    num_params = count_parameters(model)
    print(f"\nTotal trainable parameters: {num_params:,}")
    
    # Test all variants
    print("\nTesting all ResNet variants...")
    models = {
        'ResNet18': resnet18(),
        'ResNet34': resnet34(),
        'ResNet50': resnet50(),
        'ResNet101': resnet101(),
        'ResNet152': resnet152(),
        'ResNeXt50': resnext50_32x4d(),
        'Wide-ResNet50': wide_resnet50_2()
    }
    
    for name, model in models.items():
        params = count_parameters(model)
        with torch.no_grad():
            out = model(torch.randn(1, 3, 224, 224))
        print(f"  {name}: {params:,} parameters, output shape: {out.shape}")

if __name__ == "__main__":
    test_resnet50()

# Example of creating and using the model
"""
# Basic usage
model = ResNet50(num_classes=1000)

# For custom number of classes (e.g., CIFAR-10)
model = ResNet50(num_classes=10)

# With zero initialization of residual connections
model = ResNet50(num_classes=1000, zero_init_residual=True)

# Get model summary
model_summary(model)

# Extract features
x = torch.randn(1, 3, 224, 224)
features = model.get_feature_maps(x)
final_output = model(x)
"""