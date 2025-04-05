import os
import time

import numpy as np
import torch
import torch.nn as nn
from robustbench import load_model
import torchvision
import torchvision.transforms as transforms

from dm_wide_resnet import DMWideResNet

# Set device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Constants
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2471, 0.2435, 0.2616)
BATCH_SIZE = 1000

# Create features directory if it doesn't exist
os.makedirs('features', exist_ok=True)

# Load model
model = load_model(model_name='Wang2023Better_WRN-28-10', dataset='cifar10', threat_model='Linf').to(device)
model = model.eval()

# Initialize our model
my_model = DMWideResNet(num_classes=10,
                        depth=28,
                        width=10,
                        activation_fn=nn.SiLU,
                        mean=CIFAR10_MEAN,
                        std=CIFAR10_STD)
my_model = my_model.to(device)
my_model.load_state_dict(model.state_dict())
my_model.eval()

del model  # Free up memory

# Load CIFAR10 test dataset
transform = transforms.Compose([
    transforms.ToTensor(),
])

test_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2
)

# Extract features
features_list = []
labels_list = []

overall_start = time.time()

print(f"Extracting features from CIFAR10 test set...")
for batch_idx, (images, labels) in enumerate(test_loader):
    # Process batch
    with torch.no_grad():
        images = images.to(device)
        features = my_model(images, return_features=True)

        # Move to CPU and convert to numpy
        features_cpu = features.cpu().numpy()
        labels_cpu = labels.numpy()

        # Add to lists
        features_list.append(features_cpu)
        labels_list.append(labels_cpu)

        # Clear memory
        del images, features, features_cpu
        torch.cuda.empty_cache()

    # Print progress
    print(f"Processed batch {batch_idx + 1}/{len(test_loader)}")

# Combine all features and labels
all_features = np.vstack(features_list)
all_labels = np.concatenate(labels_list)

# Save features
save_path = 'features/cifar10_test_features.npz'
np.savez_compressed(save_path, features=all_features, labels=all_labels)

overall_end = time.time()
total_time = overall_end - overall_start
print(f"Total processing time: {total_time:.2f} seconds")
print(f"Extracted and saved features for {len(all_labels)} test images to {save_path}")
print(f"Feature shape: {all_features.shape}")
