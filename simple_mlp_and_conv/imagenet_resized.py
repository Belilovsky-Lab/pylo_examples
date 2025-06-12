import time
import os
import io


import h5py
import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader, RandomSampler, Dataset
from torchvision import transforms
import multiprocessing
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
            print("Creating the dataset instance")
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
    


class ImagenetResized(Dataset):
    def __init__(self,transform=None,split="train"):
        """
        Args:
            image_array (numpy.ndarray): The numpy array of images.
            label_array (numpy.ndarray): The numpy array of labels corresponding to the images.
        """
        # assert image_array.shape[0] == label_array.shape[0], "The number of images and labels must match."
        # self.split = split
        # self.h5_path = h5_path
        self.n_train = 1281167 
        self.n_val = 50000
        self.n_test = 100000
        # self.data = H5Data(h5_path=h5_path, num_workers=num_workers)
        
        # Convert numpy arrays to PyTorch tensors
        if split == "train":
            self.images = torch.from_numpy(H5Data._instance.data[:self.n_train]).float() / 255 #).permute(0, 3, 1, 2)
            self.labels = torch.from_numpy(H5Data._instance.labels[:self.n_train]).long()
        elif split == "val":
            self.images = torch.from_numpy(H5Data._instance.data[self.n_train:self.n_train + self.n_val]).float() / 255
            self.labels = torch.from_numpy(H5Data._instance.labels[self.n_train:self.n_train + self.n_val]).long()
        elif split == "test":
            self.images = torch.from_numpy(H5Data._instance.data[self.n_train + self.n_val:]).float() / 255
            self.labels = torch.from_numpy(H5Data._instance.labels[self.n_train + self.n_val:]).long()
        else:
            raise ValueError("Invalid split. Choose from 'train', 'val', or 'test'.")

        print(self.images.shape)
        print(self.labels.shape)
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
    
def get_dataset(args,split="train"):
    if args.dataset == 'imagenet':
        H5Data(h5_path=f'{os.environ.get("DATA_PATH")}/imagenet_32x32x3_JPEG.h5')
        normalize_mean=(0.485, 0.456, 0.406)
        normalize_std=(0.229, 0.224, 0.225)
        transform = transforms.Compose([
            transforms.Normalize(normalize_mean, normalize_std)
        ])
        
        ds = ImagenetResized(transform=transform,split=split)
        
        return ds
    else:
        raise ValueError("Invalid dataset. Choose from 'imagenet'.")
    
def get_dataloaders(args):
    if args.dataset == 'imagenet':
        train_dataset = get_dataset(args,split="train")
        val_dataset = get_dataset(args,split="val")
        
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,num_replicas=args.world_size,rank=args.local_rank,shuffle=True,seed=args.seed)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset,num_replicas=args.world_size,rank=args.local_rank,shuffle=False,seed=args.seed)
        else:
            train_sampler = RandomSampler(train_dataset)
            val_sampler = RandomSampler(val_dataset)
        train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, num_workers=args.num_workers,prefetch_factor=10,pin_memory=True,persistent_workers=False)
        val_loader = DataLoader(val_dataset, sampler=val_sampler, batch_size=args.batch_size, num_workers=args.num_workers,prefetch_factor=2,pin_memory=True,persistent_workers=False)
        return train_loader, val_loader
    else:
        raise ValueError("Invalid dataset. Choose from 'imagenet'.")