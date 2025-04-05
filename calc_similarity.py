import os
import time

import numpy as np
import torch

# Set device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Load CIFAR-10 test features
cifar10_test_data = np.load('features/cifar10_test_features.npz')
cifar10_test_features = torch.from_numpy(cifar10_test_data['features']).to(device)
cifar10_test_labels = cifar10_test_data['labels']
num_test_images = cifar10_test_features.shape[0]

# Normalize CIFAR-10 test features
cifar10_test_features = cifar10_test_features / torch.norm(cifar10_test_features, dim=1, keepdim=True)

# Constants
FEATURES_DIR = 'features'
NUM_20M_FILES = 100  # Number of feature files for the 20M images
SAVE_INTERVAL = 200000
TOP_K = 20


# Function to calculate cosine similarity and find top-k matches
def find_top_k_matches(test_features, large_dataset_features, top_k=5):
    """
    Calculates cosine similarity and finds the top-k matches for each test feature.

    Args:
        test_features: Normalized features of the test set (tensor on GPU).
        large_dataset_features:  Features of a chunk of the large dataset (tensor on GPU).
        top_k: The number of top matches to retrieve.

    Returns:
        A tuple containing:
        - top_k_indices: Indices of the top-k matches within the chunk.
        - top_k_similarities: Corresponding cosine similarity scores.
    """
    # Calculate cosine similarity (dot product since features are normalized)
    similarities = torch.matmul(test_features, large_dataset_features.T)

    # Get top-k similarities and indices
    top_k_similarities, top_k_indices = torch.topk(similarities, top_k, dim=1)

    return top_k_indices, top_k_similarities


# Initialize lists to store results
all_top_k_indices = []
all_top_k_similarities = []
all_top_k_global_indices = []  # Store global indices

# Process each chunk of the 20M dataset
start_time = time.time()
for i in range(1, NUM_20M_FILES + 1):
    file_path = os.path.join(FEATURES_DIR, f'features_{i}.npz')
    print(f"Loading features from {file_path}")

    # Load features chunk
    data = np.load(file_path)
    features = torch.from_numpy(data['features']).to(device)
    # labels = data['labels']  #  Not used in similarity calculation, but could be used for verification

    # Normalize features
    features = features / torch.norm(features, dim=1, keepdim=True)

    # Find top-k matches for the current chunk
    top_k_indices, top_k_similarities = find_top_k_matches(cifar10_test_features, features, TOP_K)

    # Convert local indices to global indices
    top_k_global_indices = top_k_indices + (i - 1) * SAVE_INTERVAL

    # Append to the lists
    all_top_k_indices.append(top_k_indices.cpu().numpy())
    all_top_k_similarities.append(top_k_similarities.cpu().numpy())
    all_top_k_global_indices.append(top_k_global_indices.cpu().numpy())

    # Clean up
    del data, features, top_k_indices, top_k_similarities, top_k_global_indices
    torch.cuda.empty_cache()
    print(f"Processed chunk {i}/{NUM_20M_FILES}")
    print(f"Time taken: {time.time() - start_time:.2f} seconds")

end_time = time.time()
print(f"Time taken for similarity calculation: {end_time - start_time:.2f} seconds")

# Combine results from all chunks
all_top_k_indices = np.concatenate(all_top_k_indices, axis=1)
all_top_k_similarities = np.concatenate(all_top_k_similarities, axis=1)
all_top_k_global_indices = np.concatenate(all_top_k_global_indices, axis=1)

# Find the overall top-k matches across all chunks
final_top_k_indices = np.argsort(-all_top_k_similarities, axis=1)[:, :TOP_K]
final_top_k_similarities = np.take_along_axis(all_top_k_similarities, final_top_k_indices, axis=1)
final_top_k_global_indices = np.take_along_axis(all_top_k_global_indices, final_top_k_indices, axis=1)

# Print results and save them
results = []
for i in range(num_test_images):
    print(f"Test Image {i} (Label: {cifar10_test_labels[i]}):")
    image_results = []
    for j in range(TOP_K):
        global_index = final_top_k_global_indices[i, j]
        similarity = final_top_k_similarities[i, j]
        file_number = (global_index // SAVE_INTERVAL) + 1
        local_index = global_index % SAVE_INTERVAL
        print(
            f"  Match {j + 1}: File features_{file_number}.npz, Index: {local_index}, Similarity: {similarity:.4f}, Global Index: {global_index}"
        )
        image_results.append({
            'file': f'features_{file_number}.npz',
            'index': int(local_index),  # Convert to standard Python int
            'similarity': float(similarity),  # Convert to standard Python float
            'global_index': int(global_index)  # Convert to standard Python int
        })
    results.append(image_results)

np.savez_compressed('similarity_results_top20.npz', results=results)
