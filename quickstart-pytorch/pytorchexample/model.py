from __future__ import annotations
from typing import List, Tuple
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

class CustomFashionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Passes the image through the layers defined in __init__
        return self.net(x)

    def train_epoch(
        self, 
        train_loader: DataLoader, 
        criterion: nn.Module, 
        optimizer: torch.optim.Optimizer, 
        device: torch.device,
        global_model: nn.Module = None,
        mu: float = 0.0,
        c_global: List[torch.Tensor] = None, # 🏗️ SCAFFOLD: Server's compass
        c_local: List[torch.Tensor] = None   # 🏗️ SCAFFOLD: Client's compass
    ) -> Tuple[float, float]:
        self.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = self.forward(images)
            loss = criterion(outputs, labels)
            
            # --- FEDPROX LOGIC ---
            if global_model is not None and mu > 0.0:
                proximal_term = 0.0
                for local_param, global_param in zip(self.parameters(), global_model.parameters()):
                    proximal_term += torch.square(local_param - global_param).sum()
                loss += (mu / 2.0) * proximal_term
            # ---------------------
            
            loss.backward()
            
            # ==========================================
            # 🏗️ SCAFFOLD LOGIC (Gradient Correction) 🏗️
            # ==========================================
            if c_global is not None and c_local is not None:
                with torch.no_grad():
                    for param, c, c_k in zip(self.parameters(), c_global, c_local):
                        if param.grad is not None:
                            # Move compasses to correct device
                            c_tensor = c.to(device)
                            c_k_tensor = c_k.to(device)
                            
                            # grad = grad - c_k + c
                            param.grad.data.add_(c_tensor - c_k_tensor)
            # ==========================================
            
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        avg_loss = running_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

    @torch.no_grad()
    def test_epoch(
        self, 
        test_loader: DataLoader, 
        criterion: nn.Module, 
        device: torch.device
    ) -> Tuple[float, float]:
        self.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = self.forward(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        avg_loss = running_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

    def get_model_parameters(self) -> List[np.ndarray]:
        # Converts PyTorch tensors to NumPy arrays for Flower to transmit
        return [val.cpu().numpy() for _, val in self.state_dict().items()]

    def set_model_parameters(self, params: List[np.ndarray]) -> None:
        # Takes NumPy arrays from the server and loads them back into the PyTorch model
        params_dict = zip(self.state_dict().keys(), params)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.load_state_dict(state_dict, strict=True)

def train_one_step(net, trainloader, optimizer, device):
    """FedSGD: Perform exactly one optimization step on one randomly sampled mini-batch."""
    net.train()
    
    # Grab exactly ONE batch from the shuffled loader
    try:
        batch = next(iter(trainloader))
    except StopIteration:
        return 0.0, 0.0, 0 
        
    images, labels = batch["image"].to(device), batch["label"].to(device)
        
    # Forward pass, loss, backward pass, and single step
    optimizer.zero_grad()
    outputs = net(images)
    loss = torch.nn.functional.cross_entropy(outputs, labels)
    loss.backward()
    optimizer.step()
    
    # Calculate accuracy for this batch
    correct = (torch.argmax(outputs, dim=1) == labels).sum().item()
    accuracy = correct / len(labels)
    
    return loss.item(), accuracy, len(labels)