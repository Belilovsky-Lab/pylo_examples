
import torch
import numpy as np
import random
import time
import os
import math
import h5py
import multiprocessing
from torch.utils.data import Dataset, RandomSampler
from torchvision import datasets, transforms
from PIL import Image
import math

import io





def set_torch_seed(seed = None, deterministic: bool = False):
    """
    Set all relevant seeds for PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
        deterministic (bool): If True, sets deterministic behavior for cuDNN (may slow down performance).
    """
    if seed is None:
        seed = int(time.time())

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU

    # Ensure deterministic behavior in CuDNN (at the cost of performance)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic  # If False, may slow down training

    # Ensuring reproducibility in distributed training
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" if deterministic else ""

    print(f"PyTorch seed set to {seed}. Deterministic: {deterministic}")




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




class Timer_dataset:
    def __init__(self, func):
        self.func = func
    
    def __call__(self, *args, **kwargs):
        start_time = time.time()
        result = self.func(*args, **kwargs)
        end_time = time.time()
        print(f"Executing {self.func.__name__} took {end_time - start_time:.4f} seconds.")
        return result


def process_batch(encoded_images):
    """Process a batch of encoded images into numpy arrays."""
    return [np.array(Image.open(io.BytesIO(img_data)).convert('RGB')) for img_data in encoded_images]

class H5Data:
    _instance = None

    @Timer_dataset
    def __new__(cls, h5_path, num_workers=8):
        if cls._instance is None:
            print("Loading dataset into memory")
            cls._instance = super(H5Data, cls).__new__(cls)
            
            # Read the encoded images and labels from the H5 file
            with h5py.File(h5_path, 'r') as file:
                encoded_images = file['encoded_images'][:]
                targets = file['targets'][:]
            
            # Determine the number of workers
            if num_workers is None:
                num_workers = multiprocessing.cpu_count()
            
            # Create batches of encoded images
            batch_size = len(encoded_images) // num_workers
            image_batches = [encoded_images[i:i + batch_size] for i in range(0, len(encoded_images), batch_size)]

            # Use multiprocessing to process the batches
            with multiprocessing.Pool(num_workers) as pool:
                image_arrays = pool.map(process_batch, image_batches)
            
            # Flatten the list of lists to a single list
            cls._instance.data = np.array([img for sublist in image_arrays for img in sublist])
            cls._instance.labels = np.squeeze(targets)

        return cls._instance

class NumpyImageDataset(Dataset):
    def __init__(self,transform=None):
        """
        Args:
            image_array (numpy.ndarray): The numpy array of images.
            label_array (numpy.ndarray): The numpy array of labels corresponding to the images.
        """
        # assert image_array.shape[0] == label_array.shape[0], "The number of images and labels must match."
        # self.split = split
        # self.h5_path = h5_path
        self.n_train = 1024933 #1281167
        self.n_val = 50000
        self.n_test = 100000
        # self.data = H5Data(h5_path=h5_path, num_workers=num_workers)
        
        # Convert numpy arrays to PyTorch tensors
        self.images = torch.from_numpy(H5Data._instance.data[:self.n_train]).float() / 255 #).permute(0, 3, 1, 2)
        self.labels = torch.from_numpy(H5Data._instance.labels[:self.n_train]).long()

        print("Image tensor shape:",self.images.shape)
        print("Label tensor shape:",self.labels.shape)
        self.transform = transform
    
    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image, label = self.images[idx], self.labels[idx]
        # print('image',image.shape)
        if self.transform:
            image = self.transform(image.permute(2,1,0))
        
        # print(image.shape)
        return image, label




def generate_data(num_samples=10000, num_classes=1000, img_shape=(64, 64, 3), seed=42):
    np.random.seed(seed)
    
    # Generate class labels
    labels = np.random.randint(0, num_classes, size=num_samples).astype(np.int64)
    
    # Generate class-specific means (each class has a unique mean)
    class_means = np.linspace(0, 255, num_classes).reshape(-1, 1, 1, 1)
    
    # Generate images with the corresponding mean for each class
    images = np.zeros((num_samples, *img_shape), dtype=np.uint8)
    for i in range(num_samples):
        class_mean = class_means[labels[i]]
        images[i] = np.clip(np.random.normal(loc=class_mean, scale=30, size=img_shape), 0, 255).astype(np.float32)
    
    return labels, images


class RandomNumpyImageDataset(Dataset):
    def __init__(self,transform=None):
        """
        Args:
            image_array (numpy.ndarray): The numpy array of images.
            label_array (numpy.ndarray): The numpy array of labels corresponding to the images.
        """
        self.transform = transform
        self.labels, self.images =  generate_data()
        print(self.images.shape)
        print(self.labels.shape)
    
    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image, label = self.images[idx], self.labels[idx]
        return self.transform(image), label



def get_dataset(args):

    if args.dataset_name == 'imagenet':
        H5Data(h5_path=f'{args.data_dir}/imagenet_{args.image_size}x{args.image_size}x3_JPEG.h5')

        normalize_mean=(0.485, 0.456, 0.406)
        normalize_std=(0.229, 0.224, 0.225)
        
        transform = transforms.Compose([
            transforms.Normalize(normalize_mean, normalize_std)
        ])

        trainset = NumpyImageDataset(transform)

        random_sampler = RandomSampler(
            trainset, 
            replacement=True,
            num_samples=args.num_iters * 4096 * args.num_trials
        )

        train_loader = torch.utils.data.DataLoader(
            trainset, 
            sampler=random_sampler, 
            batch_size=args.batch_size,
            shuffle=not args.no_shuffle, 
            drop_last=True,
            num_workers=args.train_num_workers,
            prefetch_factor=args.train_prefetch_factor,
            pin_memory=True,
            persistent_workers=False
        )
        test_loader = torch.utils.data.DataLoader(
            trainset, 
            batch_size=4096, 
            shuffle=False, 
            drop_last=True,
            num_workers=args.test_num_workers,
            prefetch_factor=args.test_prefetch_factor,
            pin_memory=True,
            persistent_workers=False
        )

    elif args.dataset_name == 'cifar10':

        
        
        transform = transforms.Compose(
                [transforms.ToTensor(),]
        )

        trainset = datasets.CIFAR10(root=args.data_dir, train=True,
                                                download=True, transform=transform)
        random_sampler = RandomSampler(
            trainset, 
            replacement=True,
            num_samples=args.num_iters * 4096 * args.num_trials
        )
        train_loader = torch.utils.data.DataLoader(trainset, 
                                                   batch_size=args.batch_size,
                                                   sampler=random_sampler, 
                                                   prefetch_factor=args.train_prefetch_factor,
                                                    shuffle=not args.no_shuffle, 
                                                    num_workers=args.train_num_workers,
                                                    pin_memory=True)

        testset = datasets.CIFAR10(
            root=args.data_dir, 
            train=False, 
            download=True, 
            transform=transform
        )
        test_loader = torch.utils.data.DataLoader(
            testset, 
            batch_size=args.batch_size,
            shuffle=False, 
            num_workers=args.test_num_workers,
            prefetch_factor=args.test_prefetch_factor,
            pin_memory=True
        )    
    elif args.dataset_name == 'random':
        
        transform = transforms.Compose(
                [transforms.ToTensor(),]
        )
        trainset = RandomNumpyImageDataset(transform)

        random_sampler = RandomSampler(
            trainset, 
            replacement=True,
            num_samples=args.num_iters * args.batch_size * args.num_trials
        )

        train_loader = torch.utils.data.DataLoader(
            trainset, 
            sampler=random_sampler, 
            batch_size=args.batch_size,
            shuffle=not args.no_shuffle, 
            drop_last=True,
            num_workers=8,
            prefetch_factor=10,
            pin_memory=True,
            persistent_workers=False
        )
        test_loader = torch.utils.data.DataLoader(
            trainset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            drop_last=True,
            num_workers=8,
            prefetch_factor=2,
            pin_memory=True,
            persistent_workers=False
        )
    else:
        raise ValueError('Invalid dataset name')
    
    return train_loader, test_loader