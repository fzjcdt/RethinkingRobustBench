import os
import time

import numpy as np
import torch
import torch.nn as nn
from robustbench import load_model

from dm_wide_resnet import DMWideResNet

# Set device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Load model
model = load_model(model_name='Wang2023Better_WRN-28-10', dataset='cifar10', threat_model='Linf').to(device)
model = model.eval()

# Constants
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2471, 0.2435, 0.2616)
BATCH_SIZE = 1000
SAVE_INTERVAL = 200000  # Save features every 200k images
DATA_FILES = ['20m_part1.npz', '20m_part2.npz']

# Create features directory if it doesn't exist
os.makedirs('features', exist_ok=True)

# Initialize model
my_model = DMWideResNet(num_classes=10,
                        depth=28,
                        width=10,
                        activation_fn=nn.SiLU,
                        mean=CIFAR10_MEAN,
                        std=CIFAR10_STD)
my_model = my_model.to(device)
my_model.load_state_dict(model.state_dict())
my_model.eval()

del model

# Process each data file
global_idx = 0
batch_features = []
batch_labels = []

overall_start = time.time()
interval_start = time.time()

for file_idx, file_name in enumerate(DATA_FILES):
    print(f"Processing file {file_idx + 1}/{len(DATA_FILES)}: {file_name}")

    # Load data in chunks to reduce memory usage
    data = np.load(file_name, mmap_mode='r')  # Use memory mapping to reduce RAM usage
    images = data['image']
    labels = data['label']

    num_images = len(images)
    print(f"Number of images: {num_images}")
    num_batches = (num_images + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min((batch_idx + 1) * BATCH_SIZE, num_images)

        # Load only the current batch into memory
        batch_images = images[start_idx:end_idx]
        batch_image_labels = labels[start_idx:end_idx]

        # Convert to tensors
        images_tensor = torch.from_numpy(batch_images.transpose(0, 3, 1, 2)).float() / 255.0
        labels_tensor = torch.from_numpy(batch_image_labels).long()

        # Clear memory
        del batch_images, batch_image_labels

        # Process batch
        with torch.no_grad():
            images_tensor = images_tensor.to(device)
            outputs = my_model(images_tensor, return_features=True)

            # Move features to CPU and convert to numpy to save memory on GPU
            outputs_cpu = outputs.cpu().numpy()
            labels_cpu = labels_tensor.numpy()

            # Append to batch lists
            batch_features.append(outputs_cpu)
            batch_labels.append(labels_cpu)

            # Clear memory
            del images_tensor, outputs, outputs_cpu
            torch.cuda.empty_cache()

        # Update global index
        global_idx += (end_idx - start_idx)

        # Save features if we've processed enough images
        if global_idx % SAVE_INTERVAL == 0 or (file_idx == len(DATA_FILES) - 1 and batch_idx == num_batches - 1):
            # Combine batches
            features_array = np.vstack(batch_features)
            labels_array = np.concatenate(batch_labels)

            # Save features
            save_path = f'features/features_{global_idx // SAVE_INTERVAL}.npz'
            np.savez_compressed(save_path, features=features_array, labels=labels_array)

            # Print progress and time
            interval_end = time.time()
            elapsed = interval_end - interval_start
            print(f"Processed {global_idx} images. Saved features to {save_path}")
            print(f"Time for last {len(features_array)} images: {elapsed:.2f} seconds")
            print()

            # Reset batch lists and timer
            batch_features = []
            batch_labels = []
            interval_start = time.time()

overall_end = time.time()
total_time = overall_end - overall_start
print(f"Total processing time: {total_time:.2f} seconds for {global_idx} images")
