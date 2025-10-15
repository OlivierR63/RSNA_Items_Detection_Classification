
import tensorflow as tf
import pytest

# Ensure the import path is correct for your project structure
from src.core.data_handlers.dicom_dataset import DicomTFDataset

# NOTE: The base_dir should be replaced with a path accessible in the testing environment (e.g., relative path)
# Using 'pathlib' or 'os.path.join' is generally safer for cross-platform compatibility.
# For this example, we keep the original structure for reference but acknowledge the risk.

def test_dicom_tf_dataset(dicom_samples_root):
    """Test the DicomTFDataset class to ensure it correctly loads DICOM files 
    and handles dynamic shapes via padded_batch.
    
    This test verifies that:
    - The expected directory exists and contains DICOM files.
    - The dataset is not empty and can be iterated over.
    - The output is a tuple (image_batch, shape_batch).
    - The batch shapes are consistent with batch size and padded dimensions.
    - No significant loading errors (indicated by all -1.0 values) occur.
    """
    
    # --- Setup ---
    # The path is injected by the pytest fixture
    dicom_dir = dicom_samples_root
    BATCH_SIZE = 2
    
    # Check that the directory exists and contains DICOM files
    assert tf.io.gfile.exists(dicom_dir), f"Directory {dicom_dir} does not exist"
    dicom_files = tf.io.gfile.glob(f"{dicom_dir}/*/*/*.dcm")
    assert len(dicom_files) >= 1, "No DICOM files found in the test fixture directory"

    # Create the dataset
    dataset = DicomTFDataset(root_dir=dicom_dir)
    tf_dataset = dataset.create_tf_dataset(batch_size=BATCH_SIZE)

    # Check that the dataset is not None
    assert tf_dataset is not None, "Dataset is None"

    # --- Iteration and Assertion ---
    count = 0
    # The dataset now yields a tuple: (image_batch, shape_batch)
    for image_batch, shape_batch in tf_dataset:
        count += 1
        
        # Check the first batch structure and shape
        if count >= 1:
            # 1. Check Image Batch Shape
            # Expected shape: (BATCH_SIZE, PADDED_HEIGHT, PADDED_WIDTH, PADDED_DEPTH)
            assert len(image_batch.shape) == 4, f"Image batch expected 4D tensor, got shape {image_batch.shape}"
            assert image_batch.shape[0] == BATCH_SIZE, f"Expected batch size {BATCH_SIZE}, got {image_batch.shape[0]}"
            
            # Padded dimensions (1, 2, 3) are dynamic (None in padded_shapes)
            assert image_batch.shape[1] is not None and image_batch.shape[2] is not None, "Padded image dimensions should be known after batching."

            # 2. Check Shape Batch Shape
            # Expected shape: (BATCH_SIZE, 3) -> [H, W, D/C] for each image
            assert len(shape_batch.shape) == 2, f"Shape batch expected 2D tensor, got shape {shape_batch.shape}"
            assert shape_batch.shape[0] == BATCH_SIZE, f"Shape batch expected batch size {BATCH_SIZE}, got {shape_batch.shape[0]}"
            assert shape_batch.shape[1] == 3, f"Shape vector expected size 3, got {shape_batch.shape[1]}"
            
            # 3. Check for loading errors
            # Verify that not all values in the image batch are the error code (-1.0).
            # We check if the sum of all elements is not equal to -1.0 * total elements.
            total_elements = tf.cast(tf.reduce_prod(tf.shape(image_batch)), tf.float32)
            error_sum = total_elements * -1.0
            
            # Check if the sum of the image batch is NOT equal to the error sum
            # We use a small epsilon for floating point comparison robustness
            assert tf.abs(tf.reduce_sum(image_batch) - error_sum) > 1e-6, "All values in image batch are -1.0, indicating an error in loading DICOM files."

            break # Stop after checking the first batch
