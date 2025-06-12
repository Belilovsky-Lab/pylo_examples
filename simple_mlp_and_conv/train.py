'''Train CIFAR10 with PyTorch.'''
from numpy import dtype
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import wandb
import csv
from datetime import datetime


from utils import progress_bar


parser = argparse.ArgumentParser(description='PyTorch Model Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument("--weight_decay", type=float, default=5e-4, help="weight decay")
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument("--data_dir", type=str, default="./data", help="directory to store dataset")

parser.add_argument("--epochs", type=int, default=100, help="number of epochs to train")
parser.add_argument("--batch_size", type=int, default=128, help="batch size")
parser.add_argument("--seed", type=int, default=42, help="random seed")
parser.add_argument("--model", type=str, default="mlp", help="model architecture")
parser.add_argument("--opt", type=str, default="velo", help="optimizer")

parser.add_argument("--dataset", type=str, default="cifar10", help="dataset")
parser.add_argument('--compile', action='store_true', help='compile model with torch.compile')
parser.add_argument('--amp', action='store_true', help='use automatic mixed precision')

args = parser.parse_args()

# Initialize wandb
wandb.init(project="pylo-examples", config=vars(args))

# Initialize CSV logging
csv_file = "log.csv"
csv_fieldnames = ['epoch', 'step', 'train/loss', 'train/accuracy', 'val/loss', 'val/accuracy', 'timestamp']
with open(csv_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
    writer.writeheader()

def log_to_csv(data):
    with open(csv_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
        writer.writerow(data)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

if args.dataset == "cifar10": 
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=4)
elif args.dataset == "cifar100":
    trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=4)
elif args.dataset == "imagenet32x32":
    from imagenet_resized import ImagenetResized, H5Data
    H5Data(h5_path=f'{args.data_dir}/imagenet_32x32x3_JPEG.h5')
    normalize_mean=(0.485, 0.456, 0.406)
    normalize_std=(0.229, 0.224, 0.225)
    transform_override = transforms.Compose([
        transforms.Normalize(normalize_mean, normalize_std)
    ])
    trainset = ImagenetResized(split="train", transform=transform_override)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    testset = ImagenetResized(split="val", transform=transform_override)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=4)
else:
    raise ValueError("Invalid dataset. Choose from 'cifar10', 'cifar100', or 'imagenet32x32'.")

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

if args.dataset == "imagenet32x32":
    classes = [f"imagenet_{i}" for i in range(1000)]  # Adjust for ImageNet classes
    num_classes = 1000
elif args.dataset == "cifar10":
    num_classes = 10
elif args.dataset == "cifar100":
    num_classes = 100

# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet18()
if "mu" not in args.opt:
    if args.model == "mlp":
        from mlp import MLP
        net = MLP(in_channels=3*32*32, width=128, num_classes=num_classes, nonlin=F.relu, bias=True)
        net.reset_parameters()
    elif args.model == "resnet18":
        from resnet import ResNet18
        net = ResNet18(num_classes=num_classes)
    elif args.model == "resnet50":
        from resnet import ResNet50
        net = ResNet50(num_classes=num_classes)
else:
    if args.model == "mlp":
        from mumlp import MuMLP
        from mup import set_base_shapes
        net = MuMLP(3*32*32, width=128,num_classes=num_classes)
        base_model = MuMLP(3*32*32, width=1, num_classes=num_classes)
        delta_model = MuMLP(3*32*32,width=2, num_classes=num_classes)
        set_base_shapes(net, base_model, delta=delta_model, override_base_dim=1)
        net.reset_parameters()
        
    elif args.model == "resnet50":
        from mu_resnet import mu_resnet50
        from mup import set_base_shapes
        net = mu_resnet50(width=1, num_classes=num_classes)
        base_model = mu_resnet50(width=1, num_classes=num_classes)
        delta_model = mu_resnet50(width=2, num_classes=num_classes)
        set_base_shapes(net, base_model, delta=delta_model, override_base_dim=1)
        net.reset_parameters_all()
    elif args.model == "resnet18":
        from mu_resnet import mu_resnet18
        from mup import set_base_shapes
        net = mu_resnet18(width=1, num_classes=num_classes)
        base_model = mu_resnet18(width=1, num_classes=num_classes)
        delta_model = mu_resnet18(width=2, num_classes=num_classes)
        set_base_shapes(net, base_model, delta=delta_model, override_base_dim=1)
        net.reset_parameters_all()
        

net = net.to(device)
if device == 'cuda':
    # net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()

if args.opt == "velo":
    print("using velo optimizer")
    from pylo.optim import VeLO
    optimizer = VeLO(net.parameters(), lr=args.lr,num_steps=150_000,weight_decay=args.weight_decay)
    # Create a fake scheduler that doesn't modify learning rate
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
elif args.opt == "mulo":
    print("using MuLO optimizer")
    from pylo.optim import MuLO_CUDA
    optimizer = MuLO_CUDA(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Create a fake scheduler that doesn't modify learning rate
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
elif args.opt == "adamw":
    print("using AdamW optimizer")
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
else:
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# Initialize GradScaler for mixed precision training
scaler = torch.cuda.amp.GradScaler() if args.amp and device == 'cuda' else None
if args.amp and device == 'cuda':
    print("==> Using Automatic Mixed Precision (AMP)")

# Compile model for faster training
if args.compile:
    print("==> Compiling model with torch.compile")
    net = torch.compile(net)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        
        # Use autocast for mixed precision
        if args.amp and device == 'cuda':
            with torch.amp.autocast("cuda",dtype=torch.bfloat16):
                outputs = net(inputs)
                loss = criterion(outputs, targets)
            
            # Scale loss and backward pass
            scaler.scale(loss).backward()
            
            if args.opt == "velo":
                scaler.step(optimizer, loss=loss)
                scaler.update()
            else:
                scaler.step(optimizer)
                scaler.update()
        else:
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            if args.opt == "velo":
                optimizer.step(loss=loss)
            else:
                optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        wandb.log({
            "progress/loss": train_loss/(batch_idx+1),
            "progress/accuracy": 100.*correct/total,
            "epoch": epoch,
            "step": batch_idx + epoch * len(trainloader)
        })

    # Log training metrics
    train_acc = 100.*correct/total
    avg_train_loss = train_loss/len(trainloader)
    
    wandb.log({
        "train/loss": avg_train_loss,
        "train/accuracy": train_acc,
        "epoch": epoch
    })
    
    return avg_train_loss, train_acc


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Use autocast for mixed precision inference
            if args.amp and device == 'cuda':
                with torch.amp.autocast("cuda",dtype=torch.bfloat16):
                    outputs = net(inputs)
                    loss = criterion(outputs, targets)
            else:
                outputs = net(inputs)
                loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    avg_test_loss = test_loss/len(testloader)
    
    # Log validation metrics
    wandb.log({
        "val/loss": avg_test_loss,
        "val/accuracy": acc,
        "epoch": epoch
    })
    
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc
    
    return avg_test_loss, acc

epochs = 150_000 / (len(trainloader))  
for epoch in range(start_epoch, int(epochs)):
    train_loss, train_acc = train(epoch)
    val_loss, val_acc = test(epoch)
    scheduler.step()
    
    # Log to CSV
    log_data = {
        'epoch': epoch,
        'step': epoch,
        'train/loss': train_loss,
        'train/accuracy': train_acc,
        'val/loss': val_loss,
        'val/accuracy': val_acc,
        'timestamp': datetime.now().isoformat()
    }
    log_to_csv(log_data)

wandb.finish()
