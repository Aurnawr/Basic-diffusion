import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import tqdm
#Dataset preprocessing

batch_size = 4

#defining transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) #mean and std for every channel 
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)


trainloader = DataLoader(trainset, batch_size=batch_size,
                         shuffle=True, num_workers=2)

testloader = DataLoader(testset, batch_size=batch_size,
                        shuffle=False, num_workers=2)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classes = trainset.classes


def denormalize(images: torch.Tensor) -> torch.Tensor:
    """Invert the CIFAR10 normalization so images are viewable."""
    mean = torch.tensor([0.5, 0.5, 0.5], device=images.device).view(1, 3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5], device=images.device).view(1, 3, 1, 1)
    return images * std + mean


def show_batch(images: torch.Tensor, labels: torch.Tensor, title: str) -> None:
    """Visualize a single mini-batch sampled from the dataloader."""
    images = images.cpu()
    labels = labels.cpu()
    grid = torchvision.utils.make_grid(denormalize(images))
    np_grid = grid.numpy().transpose(1, 2, 0)
    plt.figure(figsize=(6, 4))
    plt.imshow(np_grid)
    plt.axis('off')
    plt.title(title)
    plt.show(block=False)
    plt.pause(0.001)


num_epochs = 2
sample_every_n_steps = max(1, len(trainloader) // 4)

for epoch in range(num_epochs):
    progress_bar = tqdm.tqdm(trainloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
    for batch_idx, (images, labels) in enumerate(progress_bar):
        images = images.to(device)
        labels = labels.to(device)

        # Training step would go here (forward, loss, backward, optimizer.step()).

        if batch_idx == 0 or (batch_idx + 1) % sample_every_n_steps == 0:
            label_names = [classes[label.item()] for label in labels[:4]]
            title = f"Epoch {epoch + 1} | Batch {batch_idx + 1} | " \
                + ", ".join(label_names)
            show_batch(images[:4], labels[:4], title)


