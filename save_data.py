import numpy as np
import os

# --- Configuration ---
# List of input data files containing original images and labels (likely split)
DATA_FILES = ['20m_part1.npz', '20m_part2.npz']
# File containing pre-computed similarity results (e.g., top K similar items for test items)
SIMILARITY_RESULTS_FILE = 'similarity_results_top20.npz'
# Directory where the output file will be saved
OUTPUT_DIR = './'  # Or the directory where you want to save the output file
# Name of the output file
OUTPUT_FILE = '20m_top20.npz'


def extract_and_save_similar_images_with_info(data_files, similarity_results_file, output_dir):
    """
    Extracts images highly similar to test data based on pre-computed results
    and saves them into a new data file. The new file includes the extracted
    image, its label, the ID of the corresponding test sample, the similarity rank,
    and the similarity value.

    Args:
        data_files (list): List of data file paths containing original images and labels.
                           Assumes data might be split across these files.
        similarity_results_file (str): File path containing similarity results.
                                       Expected format: .npz file with a 'results' array,
                                       where each element corresponds to a test image and
                                       contains a list of {'global_index': idx, 'similarity': score} dicts
                                       for its top similar images.
        output_dir (str): Directory where the new aggregated data file will be saved.
    """

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Load similarity results
    print(f"Loading similarity results from: {similarity_results_file}")
    # allow_pickle=True might be needed if the results contain complex objects,
    # but be cautious about loading pickled data from untrusted sources.
    loaded_results = np.load(similarity_results_file, allow_pickle=True)['results']
    print(f"Loaded results for {len(loaded_results)} test images.")

    # Lists to store the extracted data
    all_images = []
    all_labels = []
    all_test_image_ids = []
    all_ranks = []
    all_similarities = []

    # Preload data files using memory mapping for potentially large files
    # Note: This assumes a specific split point (10,000,000) based on the logic below.
    # Consider a more general approach if the split points or number of files vary.
    print(f"Loading data file: {data_files[0]}")
    data1 = np.load(data_files[0], mmap_mode='r')
    image1 = data1['image']
    label1 = data1['label']
    print(f"Loading data file: {data_files[1]}")
    data2 = np.load(data_files[1], mmap_mode='r')
    image2 = data2['image']
    label2 = data2['label']
    print("Data files loaded.")

    # Define the split point for global index (adjust if necessary)
    SPLIT_INDEX = 10000000

    # Iterate through similarity results (one entry per test image)
    for test_image_idx, similar_images in enumerate(loaded_results):
        if (test_image_idx + 1) % 100 == 0: # Print progress periodically
             print(f"Processing test image {test_image_idx + 1}/{len(loaded_results)}")

        # Iterate through the top similar images for the current test image
        for rank, similar_image_info in enumerate(similar_images):
            global_index = similar_image_info['global_index']
            similarity_score = similar_image_info['similarity']  # Get the similarity score

            # Determine which file and index within the file the global_index corresponds to
            if global_index < SPLIT_INDEX:
                file_index = 0
                index_in_file = global_index
            else:
                file_index = 1
                index_in_file = global_index - SPLIT_INDEX

            # Retrieve the image and label from the corresponding loaded data
            try:
                if file_index == 0:
                    image = image1[index_in_file]
                    label = label1[index_in_file]
                else:
                    image = image2[index_in_file]
                    label = label2[index_in_file]

                # Append the extracted data and metadata to the lists
                all_images.append(image)
                all_labels.append(label)
                all_test_image_ids.append(test_image_idx)  # Save the ID/index of the test image
                all_ranks.append(rank)                     # Save the rank (0-based) of this similar image
                all_similarities.append(similarity_score)  # Save the similarity score
            except IndexError:
                print(f"Warning: Index out of bounds. Global index: {global_index}, "
                      f"File index: {file_index}, Index in file: {index_in_file}. Skipping.")
            except KeyError as e:
                print(f"Warning: Key error {e} for global index {global_index}. Skipping.")


    # It's good practice to close memory-mapped files, although Python might handle it at exit
    # data1.close() # Uncomment if you want explicit closing
    # data2.close() # Uncomment if you want explicit closing

    # Convert lists to numpy arrays before saving
    print("Converting lists to NumPy arrays...")
    all_images_np = np.array(all_images)
    all_labels_np = np.array(all_labels)
    all_test_image_ids_np = np.array(all_test_image_ids)
    all_ranks_np = np.array(all_ranks)
    all_similarities_np = np.array(all_similarities)

    # Save the aggregated data into a new compressed .npz file
    output_file_path = os.path.join(output_dir, OUTPUT_FILE)
    print(f"Saving extracted data to: {output_file_path}")
    np.savez_compressed(output_file_path,
                        image=all_images_np,
                        label=all_labels_np,
                        test_image_id=all_test_image_ids_np, # ID of the test image this is similar to
                        rank=all_ranks_np,                  # Rank of similarity (e.g., 0 = most similar)
                        similarity=all_similarities_np)     # Actual similarity score

    print(f"Successfully saved extracted data with {len(all_images_np)} entries.")


# --- Main Execution ---
if __name__ == "__main__":
    print("Starting extraction process...")
    # Call the main function
    extract_and_save_similar_images_with_info(DATA_FILES, SIMILARITY_RESULTS_FILE, OUTPUT_DIR)
    print("Extraction and saving complete.")
