# coding: utf-8

import tensorflow as tf
import SimpleITK as sitk
from pathlib import Path
from typing import List, Tuple
import os


class DicomTFDataset:
    """TensorFlow Dataset for loading DICOM files, featuring optimized Python 
    enumeration and dynamic shape handling via padded_batch.
    
    The dataset returns a tuple: (DICOM image tensor, original image shape tensor).
    """

    def __init__(self, root_dir: str) -> None:
        """Initialize the dataset and discover file paths in Python (Eager mode).

        Args:
            root_dir (str): Path to the root directory where DICOM files are stored.
                            Expected structure: root_dir/study_id/series_id/*.dcm.
        """
        self._root_dir = root_dir
        if not tf.io.gfile.exists(self._root_dir):
            raise ValueError(f"Root directory {self._root_dir} does not exist.")

        # 1. Generate ALL file paths in Python using pathlib.glob.
        # This is more efficient than using tf.data.interleave for path discovery.
        self._file_paths_list = self._generate_file_paths_python()

        if not self._file_paths_list:
             raise FileNotFoundError(f"No DICOM files found in {self._root_dir} following the structure */*/*.dcm.")

        # 2. Convert the list of strings into a simple TensorFlow Dataset.
        self._file_paths_dataset = tf.data.Dataset.from_tensor_slices(self._file_paths_list)


    def _generate_file_paths_python(self) -> List[str]:
        """Recursively generates all DICOM file paths using Python's Pathlib.
        
        Assumes structure: root_dir/study_id/series_id/*.dcm
        
        Returns:
            List[str]: A list of absolute, POSIX-style paths to all DICOM files.
        """
        root_path = Path(self._root_dir)
        # Use glob to find all *.dcm files within two levels of subdirectories
        # .as_posix() ensures '/' separators, compatible with TF/Linux/Windows
        return [p.resolve().as_posix() for p in root_path.glob('*/*/*.dcm')]


    def _py_load_dicom_tf(self, path: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Python function executed by tf.py_function to load the DICOM 
        using SimpleITK, extracting both the image and its shape.
        
        Args:
            path (tf.Tensor): The DICOM file path (tf.string/bytes).

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: (The DICOM image, The image shape).
        """
        try:
            # Decode the path tensor (bytes) into a Python string
            path_str = path.numpy().decode('utf-8')
            
            # --- SimpleITK Loading ---
            img = sitk.ReadImage(path_str)
            img_array = sitk.GetArrayFromImage(img)
            
            # Convert to tensors
            image_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
            # The shape (H, W, D or C) is returned as an Int32 tensor
            shape_tensor = tf.constant(img_array.shape, dtype=tf.int32)
            
            return image_tensor, shape_tensor

        except Exception as e:
            tf.print(f"Error loading DICOM file {path_str}: {e}")
            # Return a small dummy tensor (1, 1, 1) and its shape on error 
            # to prevent the pipeline from immediately crashing during batching.
            image_tensor_dummy = tf.fill((1, 1, 1), -1.0)
            shape_tensor_dummy = tf.constant([1, 1, 1], dtype=tf.int32)
            return image_tensor_dummy, shape_tensor_dummy


    def _load_dicom(self, file_path: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """tf.py_function wrapper for the SimpleITK loading function."""
        return tf.py_function(
            self._py_load_dicom_tf, 
            [file_path], 
            (tf.float32, tf.int32) # Specify expected output types: (Image, Shape)
        )


    def create_tf_dataset(self, batch_size: int = 8) -> tf.data.Dataset:
        """Create an optimized TensorFlow Dataset, loading the image and its shape.
        Uses padded_batch to handle variable image sizes within a batch.

        Args:
            batch_size (int, optional): Number of DICOM files per batch. Defaults to 8.

        Returns:
            tf.data.Dataset: A batched and prefetched dataset of (image, shape) tuples.
        """
        
        dataset = self._file_paths_dataset
        
        # 1. Mapping (parallel loading)
        # Each element becomes a tuple: (image_tensor, shape_tensor)
        dataset = dataset.map(
            self._load_dicom,
            num_parallel_calls=tf.data.AUTOTUNE
        )

        # 2. Dynamic Batching (padded_batch)
        # This is necessary because the Image tensor has variable sizes ([None, None, None]).
        # padded_batch pads the image to the largest size in the current batch.
        dataset = dataset.padded_batch(
            batch_size=batch_size,
            padded_shapes=(
                tf.TensorShape([None, None, None]), # Image (dynamic size on all axes)
                tf.TensorShape([3])                 # Shape vector (fixed size of 3 elements)
            ),
            padding_values=(
                tf.constant(0.0, dtype=tf.float32), # Padding value for the image pixels
                tf.constant(0, dtype=tf.int32)      # Padding value for the shape vector (not used, but required)
            )
        )
        
        # 3. Prefetching for optimized throughput
        return dataset.prefetch(tf.data.AUTOTUNE)