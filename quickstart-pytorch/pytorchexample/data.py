import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split
import numpy as np
from pathlib import Path

def generate_distributed_datasets(k: int, alpha: float, save_dir: str) -> None:
    # 1. Load the FashionMNIST training set
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.FashionMNIST(root="./data_raw", train=True, download=True, transform=transform)
    
    # 2. Partition into k client subsets using Dirichlet(alpha)
    labels = np.array(dataset.targets)
    num_classes = 10
    idx_batch = [[] for _ in range(k)]
    
    for c in range(num_classes):
        idx_c = np.where(labels == c)[0]
        np.random.shuffle(idx_c)
        
        # Dirichlet distribution determines the proportions of class 'c' for each client 
        proportions = np.random.dirichlet([alpha] * k)
        proportions = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
        
        idx_split = np.split(idx_c, proportions)
        for i in range(k):
            idx_batch[i].extend(idx_split[i])

    # 3. Save one file per client into save_dir
    path = Path(save_dir)
    path.mkdir(parents=True, exist_ok=True)
    
    for i in range(k):
        # Storing indices to point to the original dataset 
        client_indices = torch.tensor(idx_batch[i])
        torch.save(client_indices, path / f"client_{i}.pt") 

def load_client_data(cid: int, data_dir: str, batch_size: int) -> tuple[DataLoader, DataLoader]:
    """
    Load a client's local dataset from a saved .pt file and split it 
    into training and validation DataLoaders.
    """
    # 1. Load the base dataset (no download needed if generate_distributed_datasets ran)
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5,), (0.5,))
    ])
    full_dataset = datasets.FashionMNIST(root="./data_raw", train=True, download=False, transform=transform)
    
    # 2. Load this specific client's indices
    client_indices = torch.load(Path(data_dir) / f"client_{cid}.pt")
    client_subset = Subset(full_dataset, client_indices)
    
    # 3. Split into Train (80%) and Validation (20%)
    train_size = int(0.8 * len(client_subset))
    val_size = len(client_subset) - train_size
    train_ds, val_ds = random_split(client_subset, [train_size, val_size])
    
    # 4. Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader