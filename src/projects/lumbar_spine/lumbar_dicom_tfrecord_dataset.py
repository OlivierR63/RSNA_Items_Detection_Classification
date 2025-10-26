# coding: utf-8

from ast import Try, TryStar
import tensorflow as tf
from typing import Dict, Tuple, List, Optional
from src.core.data_handlers.dicom_tfrecord_dataset import DicomTFRecordDataset
from src.projects.lumbar_spine.csv_metadata import CSVMetadata
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import SimpleITK as sitk
import struct
import io
from src.core.utils.logger import log_method
import logging

class LumbarDicomTFRecordDataset(DicomTFRecordDataset):
    """TensorFlow Dataset for loading DICOM TFRecords."""

    def __init__(self, config: dict, logger: Optional[logging.Logger] = None) -> None:
        # Initialize the logger first (before calling super() to log initialization)
        self._MAX_RECORDS = 25
        self._MAX_RECORDS_FLAT = self._MAX_RECORDS * 4
        self.logger = logger if logger is not None else logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing LumbarDicomTfRecordDataset")

        super().__init__(config, logger= self.logger) # Pass the logger to parent class
 

    def max_records_flat(self):
        return self._MAX_RECORDS_FLAT


    @log_method()
    def create_tf_dataset(self, batch_size: int = 8,*,
                          logger: Optional[logging.Logger] = None) -> tf.data.Dataset:
        """
        Creates an optimized TensorFlow Dataset for training or evaluation by reading 
        and processing TFRecord files asynchronously.

        The pipeline is constructed to maximize I/O and processing throughput 
        using interleave, shuffle, and prefetch mechanisms.

        Args:
            batch_size (int): The number of elements to combine into a single batch. 
                              Defaults to 8.
            logger: Automatically injected logger (optional).

        Returns:
            tf.data.Dataset: An optimized Dataset where each element is a tuple 
                             (image_tensor, metadata_dict).
        """

        logger = logger or self.logger
        logger.info(f"Creating TF Dataset with batch_size={batch_size}",
                       extra={"action": "create_dataset", "batch_size": batch_size})

        try:
            # 1. List all TFRecord files matching the pattern (e.g., 'data/*.tfrecord').
            # Shuffling the file names helps ensure better data mixing across epochs.
            tfrecord_files = tf.data.Dataset.list_files(self._tfrecord_pattern, shuffle=True)
            logger.info(f"Found {len(tfrecord_files)} TFRecord files",
                       extra={"file_count": len(tfrecord_files)})

            # 2. Use "interleave" to read data from multiple files in parallel.
            # This prevents I/O bottlenecks by having multiple readers working concurrently.
            dataset = tfrecord_files.interleave(
            
                # The map function reads a single TFRecord file and applies the parsing function.
                lambda x: tf.data.TFRecordDataset(x).map(
                    self._parse_tfrecord,
                    # Use AUTOTUNE to determine the optimal number of parallel processing threads.
                    num_parallel_calls=tf.data.AUTOTUNE
                ),
                # Interleave also uses AUTOTUNE for setting the number of concurrent calls (readers).
                num_parallel_calls=tf.data.AUTOTUNE
            )

            # 3. Apply standard Dataset optimizations.
    
            # Shuffle the processed elements in memory using a buffer.
            dataset = dataset.shuffle(buffer_size=1000)
    
            # Combine individual elements into batches for efficient model processing.
            dataset = dataset.batch(batch_size)
    
            # Prefetch data: The input pipeline fetches the next batch while the model 
            # is processing the current batch, preventing GPU/TPU starvation.
            dataset = dataset.prefetch(tf.data.AUTOTUNE)

            logger.info("Dataset pipeline created successfully",
                       extra={"status": "success"})

            return dataset

        except Exception as e:
            logger.error(f"Error creating dataset: {str(e)}", exc_info=True,
                        extra={"status": "failed", "error": str(e)})
            raise


    @log_method()
    def _generate_tfrecord_files(self, *, logger: Optional[logging.Logger] = None) -> None:
        """
        Generates TFRecord files from DICOM images and associated metadata.

        This function performs the following steps:
        1. Creates the necessary output directory for TFRecord files.
        2. Loads and merges metadata from CSV files as specified in the configuration.
        3. Encodes categorical metadata fields (e.g., condition, level, series_description, severity)
           into numerical values for compatibility with machine learning pipelines.
        4. Converts DICOM files to TFRecord format (if no existing TFRecord files are found in the output directory).

        Args:
            logger: Automatically injected logger (optional).

        Notes:
            - TFRecord files are only generated if the output directory is empty.
            - The DICOM root directory and output directory are specified in the configuration.
        """
        
        logger = logger or self.logger
        logger.info("Starting generate_tfrecord_file", extra ={"action" : "generate_tf_records"})

        try:
            # 1. Prepare the directories
            self._tfrecord_dir.mkdir(parents=True, exist_ok=True)

            # 2. Load and merge metadata
            metadata_handler = CSVMetadata(logger=logger, **self._config["csv_files"])
            logger.info("Loaded metadata from CSV files",
                        extra={"csv_files": list(self._config["csv_files"].keys())})

            # 3. Encode categorical metadata
            #    Fields affected : condition, level, series_description, severity
            encoded_metadata_df = self._encode_dataframe(metadata_handler._merged_df)
            logger.info("Encoded categorical metadata", extra={"action": "encode_metadata"})

            # 4. Convert DICOM files to TFRecords (if needed)
            if not list(self._tfrecord_dir.glob("*.tfrecord")):
                self.logger.info("  Creating TFRecord files...")
                self._convert_dicom_to_tfrecords(
                    root_dir=self._config["dicom_root_dir"],
                    metadata_df=encoded_metadata_df,
                    output_dir=str(self._tfrecord_dir)
                )
                logger.info("DICOM to TFRecord conversion completed.",
                            extra = {"status":"success"})
            else:
                logger.info("Existing TFRecords found. Skipping conversion.",
                           extra={"status": "skipped"})

        except Exception as e:
            logger.error(f"Error generating TFRecords: {str(e)}", exc_info=True,
                        extra={"status": "failed", "error": str(e)})
            raise


    def _py_deserialize_and_flatten(self, metadata_bytes_tensor: tf.Tensor) -> List[tf.Tensor]:
        """
        Deserializes and flattens metadata bytes into a list of TensorFlow tensors.

        This method takes a serialized metadata byte string (typically from a TFRecord)
        and converts it into a structured list of TensorFlow tensors. The metadata is first
        deserialized into a dictionary, then split into header tensors (scalar metadata fields)
        and a flattened tensor for records (annotations or points of interest).

        The records are padded to a fixed size (self._MAX_RECORDS_FLAT) to ensure consistent tensor shapes
        for batch processing in TensorFlow pipelines.

        Args:
            metadata_bytes_tensor (tf.Tensor): A TensorFlow tensor containing the serialized
                metadata as a byte string. This tensor is typically extracted from a TFRecord
                and represents the 'metadata' field.

        Returns:
            list[tf.Tensor]: A list of 7 TensorFlow tensors:
                - The first 6 tensors are scalar `tf.int32` tensors representing:
                    0: study_id (int)
                    1: series_id (int)
                    2: instance_number (int)
                    3: description (int, encoded category)
                    4: condition (int, encoded category)
                    5: nb_records (int, number of records/annotations)
                - The 7th tensor is a 1D `tf.float32` tensor of shape (100,) representing:
                    6: Flattened and padded records (level, severity, x, y) for up to 25 records.
                      Each record is represented by 4 values, and the tensor is padded with zeros
                      to ensure a fixed size of 100 elements (25 records * 4 values).

        Raises:
            ValueError: If the deserialized metadata structure is invalid or inconsistent.
            Exception: If the deserialization process fails (e.g., due to corrupted metadata bytes).
                      The exception will be logged and propagated to the caller.

        Notes:
            - This method is designed to be called within a `tf.py_function` context, allowing
              the use of Python code inside a TensorFlow graph.
            - The `.numpy()` call is required to convert the input tensor to a NumPy array for
              deserialization, as TensorFlow operations are not available in this context.
            - The padding ensures that the output tensor for records always has a fixed size,
              which is necessary for batching in TensorFlow datasets.
        """
        try:
            # We MUST call .numpy() here, inside the py_function boundary.
            deserialized_dict = self._deserialize_metadata(metadata_bytes_tensor.numpy())

            required_keys = ['study_id', 'series_id', 'instance_number', 'description', 'condition', 'nb_records']
            for key in required_keys:
                if key not in deserialized_dict:
                    raise ValueError(f"Missing required key in deserialized metadata: {key}")
            
             # 1. Extract header Tensors
            header_tensors = [
                    tf.constant(deserialized_dict.get('study_id', 0), dtype=tf.int32),
                    tf.constant(deserialized_dict.get('series_id', 0), dtype=tf.int32),
                    tf.constant(deserialized_dict.get('instance_number', 0), dtype=tf.int32),
                    tf.constant(deserialized_dict.get('description', 0), dtype=tf.int32),
                    tf.constant(deserialized_dict.get('condition', 0), dtype=tf.int32),
                    tf.constant(deserialized_dict.get('nb_records', 0), dtype=tf.int32)
                ]
            
             # 2. Flatten records into a 1D list of values: [level1, severity1, x1, y1, level2, severity2, x2, y2, ...]
             # If no records, create a dummy 1x4 list of zeros.
            records_list = deserialized_dict.get('records', [])
            flattened_records = [val for rec in records_list for val in rec] or [-1.0] * 4 
            
            # Pad the flattened list to the self._MAX_RECORDS_FLAT size (25 * 4 = 100 elements)          
            # Pad with a safe value (0.0)
            padded_records = flattened_records + [0.0] * (self._MAX_RECORDS_FLAT - len(flattened_records))  
            records_tensor = tf.constant(padded_records, dtype=tf.float32)
            result = header_tensors + [records_tensor]
            
            return result

        except Exception as e:
            self.logger.error(f"Error in _py_deserialize_and_flatten: {str(e)}", exc_info=True)
            # Retourne une structure par défaut
            return [
                tf.constant(0, dtype=tf.int32),  # study_id
                tf.constant(0, dtype=tf.int32),  # series_id
                tf.constant(0, dtype=tf.int32),  # instance_number
                tf.constant(0, dtype=tf.int32),  # description
                tf.constant(0, dtype=tf.int32),  # condition
                tf.constant(0, dtype=tf.int32),  # nb_records
                tf.constant([0.0] * 100, dtype=tf.float32)  # records
        ]


    @log_method()
    def _parse_tfrecord(self, example_proto: tf.Tensor,*,
                       logger: Optional[logging.Logger] = None) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """
        Parses a single TFRecord entry, extracting and processing the image and metadata.

        This function is typically used within a tf.data pipeline to transform 
        raw TFRecord strings into usable tensors and structured data.

        Args:
            example_proto (tf.Tensor): A scalar string Tensor representing one 
                                       serialized tf.train.Example protocol buffer.
            logger: Automatically injected logger (optional).

        Returns:
            Tuple[tf.Tensor, Dict]: A tuple containing the processed image tensor 
                                    and a dictionary of the deserialized metadata.
        """

        try:
            # Define the structure of the features stored in the TFRecord.
            feature_description = {
                "image": tf.io.FixedLenFeature([], tf.string),
                "metadata": tf.io.FixedLenFeature([], tf.string)
            }
    
            # Parse the scalar protocol buffer string into a dictionary of Tensors.
            example = tf.io.parse_single_example(example_proto, feature_description)

            # --- 1. Deserialize and Reshape the Image Tensor (Pure TF) ---
            image = tf.io.parse_tensor(example["image"], out_type=tf.uint16) 
            # NOTE: Using tf.uint16, the original Dicom type. Normalization to float32 happens later.
        
            # Reshape the 1D Tensor back into its original 4D shape (e.g., [Depth, Height, Width, Channels]).
            # The shape is assumed to be fixed for this dataset.
            image = tf.reshape(image, [64, 64, 64, 1])  

            # --- 2. Deserialize the Metadata (using tf.py_function) ---
            metadata_bytes = example["metadata"]
        
            # Define the output types for the metadata deserializer:
            # [0: study_id (int), 1: series_id (int), 2: instance_number (int), 3: description (int),
            #  4: condition (int), 5: nb_records (int), 6: records (list of tuples -> will be tf.float32/tf.int32)]
        
            # We call the deserializer via tf.py_function, which can execute Python code.
            # It must return a consistent set of Tensors, not a dict.
        
            # The metadata will be returned as individual scalar Tensors and one combined Tensor for records.
            # We need the inner function to return a flat list/tuple of Tensors.
        
            # To handle complex structures (the list of records), we must wrap the deserializer 
            # and parse the records list inside the tf.py_function output, returning fixed-size 
            # Tensors, which is difficult for a variable list of records.
        
            # The cleaner solution: Return only the most complex part (records) as one flattened tensor
            # and re-structure it later in the TF graph.

            # Define the output signature for tf.py_function
            # 6 scalar headers (tf.int32) + 1 padded records tensor (tf.float32)
            output_types = [tf.int32] * 6 + [tf.float32]
        
            # Call the py_function
            deserialized_tensors = tf.py_function(
                self._py_deserialize_and_flatten, 
                inp=[metadata_bytes], 
                Tout=output_types
            )
        
            # Assign the results back to named Tensors
            (study_id_t, series_id_t, instance_number_t, description_t, condition_t, nb_records_t, padded_records_t) = deserialized_tensors
        
            # Reshape the padded records tensor to (self._MAX_RECORDS, 4)
            padded_records_t = tf.reshape(padded_records_t, [25, 4])


            # --- 3. Normalize the Image (Pure TF) ---
            # The image is currently uint16. Convert to float32 and normalize.
            image = tf.cast(image, dtype=tf.float32)
            # Simple normalization: scaling pixel values to the range [0, 1].
            image = image / tf.reduce_max(image) 

            # --- 4. Return Processed Data ---
            return_object = (image, {
                "study_id": study_id_t,
                "series_id": series_id_t,
                "instance_number": instance_number_t,
                "description": description_t,
                "condition": condition_t,
                "nb_records": nb_records_t,
                # The records are now a (25, 4) float32 tensor
                "records": padded_records_t
            })

            return return_object
        
        except Exception as e:
            logger.error(f"Error parsing TFRecord: {str(e)}", exc_info=True,
                        extra={"status": "failed", "error": str(e)})
            # Return a safe dummy structure to avoid pipeline crash
            dummy_image = tf.zeros([64, 64, 64, 1], dtype=tf.float32)
            dummy_records = tf.zeros([25, 4], dtype=tf.float32)
        
            return dummy_image, {
                "study_id": tf.constant(0, dtype=tf.int32),
                "series_id": tf.constant(0, dtype=tf.int32),
                "instance_number": tf.constant(0, dtype=tf.int32),
                "description": tf.constant(0, dtype=tf.int32),
                "condition": tf.constant(0, dtype=tf.int32),
                "nb_records": tf.constant(0, dtype=tf.int32),
                "records": dummy_records
            }


    @log_method()
    def _convert_dicom_to_tfrecords(self, root_dir: str, metadata_df: pd.DataFrame,
                               output_dir: str, *,
                               logger: Optional[logging.Logger] = None) -> None:
        """
        Converts DICOM files stored in a hierarchical directory structure into
        TensorFlow TFRecord files, generating one TFRecord file per study.

        Args:
            root_dir (str): The path to the root directory containing study subfolders.
            metadata_df (pd.DataFrame): A DataFrame containing pre-processed metadata.
            output_dir (str): The directory where the resulting TFRecord files will be saved.
            logger: Automatically injected logger (optional).

        Returns:
            None: The function saves files to disk but returns nothing.
        """
        logger = logger or self.logger
        logger.info("Starting DICOM to TFRecord conversion",
                   extra={"action": "convert_dicom", "root_dir": root_dir})

        try:
            output_dir = self._setup_output_directory(output_dir)

            for study_path in tqdm(list(Path(root_dir).iterdir())):
                if not study_path.is_dir():
                    continue

                self._process_study(study_path, metadata_df, output_dir, logger)

            logger.info("DICOM to TFRecord conversion completed successfully",
                       extra={"status": "success"})

        except Exception as e:
            logger.error(f"Error during DICOM conversion: {str(e)}", exc_info=True,
                         extra={"status": "failed", "error": str(e)})
            raise

    def _setup_output_directory(self, output_dir: str) -> Path:
        """Setup the output directory, creating it if it doesn't exist."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _process_study(self, study_path: Path, metadata_df: pd.DataFrame,
                       output_dir: Path, logger: logging.Logger) -> None:
        """Process all DICOM files in a study directory and write them to a TFRecord file."""
        study_id = study_path.name
        output_path = output_dir / f"{study_id}.tfrecord"

        with tf.io.TFRecordWriter(str(output_path)) as writer:
            for series_path in study_path.iterdir():
                if not series_path.is_dir():
                    continue

                self._process_series(series_path, metadata_df, writer)

    def _process_series(self, series_path: Path, metadata_df: pd.DataFrame,
                        writer: tf.io.TFRecordWriter) -> None:
        """Process all DICOM files in a series directory and write them to the TFRecord writer."""
        series_id = int(series_path.name)

        for dicom_path in series_path.glob("*.dcm"):
            img_bytes, serialized_metadata = self._process_dicom_file(dicom_path, metadata_df)
            self._write_tfrecord_example(img_bytes, serialized_metadata, writer)

    def _process_dicom_file(self, dicom_path: Path, metadata_df: pd.DataFrame) -> Tuple[bytes, bytes]:
        """Process a single DICOM file and return serialized image and metadata."""
        img = sitk.ReadImage(str(dicom_path))
        img_array = sitk.GetArrayFromImage(img)
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.uint16)
        img_bytes = tf.io.serialize_tensor(img_tensor).numpy()

        serialized_metadata = self._get_metadata_for_file(str(dicom_path), metadata_df)

        return img_bytes, serialized_metadata

    def _write_tfrecord_example(self, img_bytes: bytes, serialized_metadata: bytes,
                                writer: tf.io.TFRecordWriter) -> None:
        """Write a single TFRecord example to the writer."""
        feature = {
            "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_bytes])),
            "metadata": tf.train.Feature(bytes_list=tf.train.BytesList(value=[serialized_metadata]))
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())



    @log_method()
    def _get_metadata_for_file(self, file_path: str, metadata_df: pd.DataFrame,*,
                              logger: Optional[logging.Logger] = None) -> dict:
        """
        Returns the serialized metadata extracted from a dataframe and associated with a given DICOM file.

        This method derives the necessary identifiers (study ID, series ID, instance number) 
        from the file path structure and uses these to query and serialize the relevant 
        records from the comprehensive metadata DataFrame.

        Args:
            file_path (str): The full path to the DICOM file (used to derive IDs).
            metadata_df (pd.DataFrame): The main DataFrame containing all metadata records.
            logger: Automatically injected logger (optional).

        Returns:
            bytes: The compact byte sequence representing the serialized metadata 
                   for the specific image, or an empty dictionary if the input DataFrame is None.
        """
        
        logger = logger or self.logger
        logger.info("Starting retrieving metadata from CSV files")
        
        try:
            if metadata_df is None:
                # NOTE: Returning bytes instead of a dict, as the caller (_convert_dicom_to_tfrecords) 
                # expects serialized bytes for the TFRecord.
                return b''

            # --- 1. Extract Identifiers from File Path ---
    
            # Assuming the file path structure is ROOT_DIR/STUDY_ID/SERIES_ID/INSTANCE_NUMBER.dcm
    
            # Split the path into parts.
            parts = Path(file_path).parts
    
            # Get the study_id (third segment from the end).
            study_id = parts[-3]
    
            # Get the series_id (second segment from the end).
            series_id = parts[-2]
    
            # Get the instance_number (file name stem, excluding the extension).
            instance_number = Path(file_path).stem

            # --- 2. Serialize Metadata ---
    
            # Call the core serialization function, passing the extracted IDs and the full DataFrame.
            # This will filter the DataFrame and create the compact byte sequence.
            serialized_metadata = self._serialize_metadata(study_id, series_id, instance_number, metadata_df)

            # Return the byte sequence.
            return serialized_metadata

            logger.info("CSV metadata retrieval completed successfully",
                       extra={"status": "success"})
        
        except Exception as e:
            logger.error(f"Error in function _get_metadata_for_file() : {str(e)}", exc_info=True,
                        extra={"status": "failed", "error": str(e)})


    @log_method()
    def _serialize_metadata(self,
                           study_id_str: str,
                           series_id_str: str,
                           instance_number_str: str,
                           data_df: pd.DataFrame,*,
                           logger: Optional[logging.Logger] = None) -> bytes:
        """
        Serializes specific metadata records from a DataFrame into a compact byte sequence.

        Args:
            study_id_str: The study identifier used to filter the DataFrame.
            series_id_str: The series identifier used to filter the DataFrame.
            instance_number_str: The name of the DICOM file, without its '.dcm' extension.
            data_df: The main DataFrame containing all metadata records.
            logger: Automatically injected logger (optional).

        Returns:
            bytes: A compact byte sequence representing the serialized metadata.
                Structure: [Header (15 bytes)] + [Record 1 (8 bytes)] + ... + [Record N (8 bytes)]

        Raises:
            ValueError: If the number of records for the given identifiers exceeds 25.
        """
        logger = logger or self.logger
        logger.info("Starting function _serialize_metadata")

        try:
            records_df = self._filter_records(data_df, study_id_str, series_id_str, instance_number_str, logger)
            if records_df.empty:
                return b''

            header_bytes = self._serialize_header(records_df, study_id_str, series_id_str, instance_number_str)
            payload_bytes = self._serialize_payload(records_df)

            metadata_bytes = header_bytes + payload_bytes

            logger.info("Function _serialize_metadata completed successfully", extra={"status": "success"})
            return metadata_bytes

        except Exception as e:
            logger.error(f"Error in function _serialize_metadata(): {str(e)}", exc_info=True,
                         extra={"status": "failed", "error": str(e)})
            raise


    def _filter_records(self, data_df: pd.DataFrame, study_id_str: str, series_id_str: str,
                        instance_number_str: str, logger: logging.Logger) -> pd.DataFrame:
        """Filters the DataFrame to get records matching the given identifiers."""
        mask = (
            (data_df["study_id"] == int(study_id_str)) &
            (data_df["series_id"] == int(series_id_str)) &
            (data_df["instance_number"] == int(instance_number_str))
        )
        records_df = data_df[mask]

        if records_df.empty:
            logger.warning(f"No metadata linked with the file {study_id_str}/{series_id_str}/{instance_number_str}.dcm")
            logger.warning("This file will not be considered during training or evaluation.")
            logger.warning("This may be due to missing or inconsistent records in the CSV files.")
            logger.warning("Please check the CSV files and ensure they contain the necessary records.")
            logger.warning("Returning an empty byte sequence for metadata")

        return records_df


    def _serialize_header(self, records_df: pd.DataFrame, study_id_str: str,
                          series_id_str: str, instance_number_str: str) -> bytes:
        """Serializes the header part of the metadata."""
        # Extract global properties (must be unique for the filtered records)
        condition = int(records_df['condition'].unique()[0])
        description = int(records_df['series_description'].unique()[0])

        # Serialize study_id and series_id as 5-byte unsigned integers (big-endian)
        study_id_bytes = int(study_id_str).to_bytes(5, byteorder='big', signed=False)
        series_id_bytes = int(series_id_str).to_bytes(5, byteorder='big', signed=False)

        # Serialize instance_number as a 2-byte unsigned short (max 9999)
        instance_number_bytes = int(instance_number_str).to_bytes(2, byteorder='big', signed=False)

        # Serialize description and condition as 1-byte unsigned characters
        description_bytes = description.to_bytes(1, byteorder='big', signed=False)
        condition_bytes = condition.to_bytes(1, byteorder='big', signed=False)

        # Serialize the number of records (max 25) as a 1-byte unsigned character
        nb_records = len(records_df)
        if nb_records > 25:
            raise ValueError("The number of records exceeds the limit of 25.")
        nb_records_bytes = nb_records.to_bytes(1, byteorder='big', signed=False)

        return study_id_bytes + series_id_bytes + instance_number_bytes + description_bytes + condition_bytes + nb_records_bytes


    def _serialize_payload(self, records_df: pd.DataFrame) -> bytes:
        """Serializes the payload part of the metadata."""
        payload_bytes = b''

        for row in records_df.itertuples():
            level = row.level
            severity = row.severity

            # Convert float coordinates to integers by scaling (multiplication by 100)
            x = round(float(row.x) * 100)
            y = round(float(row.y) * 100)

            # Serialize level and severity as 1-byte unsigned characters
            level_bytes = struct.pack('=B', level)
            severity_bytes = struct.pack('=B', severity)

            # Serialize scaled x and y as 3-byte unsigned integers (big-endian)
            x_bytes = x.to_bytes(3, byteorder='big', signed=False)
            y_bytes = y.to_bytes(3, byteorder='big', signed=False)

            payload_bytes += level_bytes + severity_bytes + x_bytes + y_bytes

        return payload_bytes


    @log_method()
    def _deserialize_metadata(self, metadata_bytes: bytes, *,
                          logger: Optional[logging.Logger] = None) -> Dict:
        """
        Deserializes a compact byte sequence back into structured metadata components.
        This function is the inverse of _serialize_metadata.
        Args:
            metadata_bytes: The byte sequence containing the serialized metadata.
            logger: Automatically injected logger (optional).
        Returns:
            dict: A dictionary containing the deserialized header values and a list of tuples
                  (level, severity, x, y) for the individual records.
        Raises:
            ValueError: If the input is not a byte sequence or if the buffer length is insufficient.
        """
        if not isinstance(metadata_bytes, bytes):
            raise ValueError("Input must be a byte sequence")

        if not metadata_bytes:
            raise ValueError("Input byte sequence is empty")

        MINIMUM_BUFFER_LENGTH = 31  # Requested minimal length (header + 1 record)
        if len(metadata_bytes) < MINIMUM_BUFFER_LENGTH:
            raise struct.error(f"Invalid buffer length: expected at least {MINIMUM_BUFFER_LENGTH} bytes, got {len(metadata_bytes)}")

        try:
            buffer = io.BytesIO(metadata_bytes)
            header = self._deserialize_header(buffer)
            records = self._deserialize_records(buffer, header['nb_records'])
            return {**header, 'records': records}
        
        except Exception as e:
            logger = logger or self.logger
            logger.error(f"Error in function _deserialize_metadata: {str(e)}", exc_info=True,
                         extra={"status": "failed", "error": str(e)})
            
            raise Exception(f"Error deserializing metadata: {str(e)}")


    def _deserialize_header(self, buffer: io.BytesIO) -> Dict:
        """Deserializes the header part of the metadata."""
        study_id = int.from_bytes(buffer.read(5), byteorder='big', signed=False)
        series_id = int.from_bytes(buffer.read(5), byteorder='big', signed=False)
        instance_number = int.from_bytes(buffer.read(2), byteorder='big', signed=False)
        description = int.from_bytes(buffer.read(1), byteorder='big', signed=False)
        condition = int.from_bytes(buffer.read(1), byteorder='big', signed=False)
        nb_records = int.from_bytes(buffer.read(1), byteorder='big', signed=False)

        return {
            'study_id': study_id,
            'series_id': series_id,
            'instance_number': instance_number,
            'description': description,
            'condition': condition,
            'nb_records': nb_records
        }


    def _deserialize_records(self, buffer: io.BytesIO, nb_records: int) -> List[Tuple]:
        """Deserializes the records part of the metadata."""
        records = []

        for _ in range(nb_records):
            level = int.from_bytes(buffer.read(1), byteorder='big', signed=False)
            severity = int.from_bytes(buffer.read(1), byteorder='big', signed=False)

            x_scaled = int.from_bytes(buffer.read(3), byteorder='big', signed=False)
            x = x_scaled / 100.0  # Rescale back to float

            y_scaled = int.from_bytes(buffer.read(3), byteorder='big', signed=False)
            y = y_scaled / 100.0  # Rescale back to float

            records.append((level, severity, x, y))

        return records



    @log_method()
    def _encode_dataframe(self, metadata_df: pd.DataFrame, *,
                            logger: Optional[logging.Logger] = None) -> pd.DataFrame:
        """
        Converts categorical textual metadata fields in a DataFrame to numerical values.
        This process is essential for preparing data for serialization or machine learning models,
        replacing descriptive strings with compact integers.

        Args:
            metadata_df: The DataFrame containing metadata to be converted.
                        Example column values: {
                                                    "condition": "Spinal Canal Stenosis",
                                                    "level": "L1-L2", ...
                                                }

        Returns:
            pd.DataFrame: The DataFrame with the specified columns converted to integer values.
        """
        logger = logger or self.logger
        logger.info("Starting function _encode_dataframe")

        try:
            # Define columns to encode and their corresponding mapping variables
            columns_to_encode = ["condition", "level", "series_description", "severity"]

            # Create mappings for each column
            mappings = self._create_mappings(metadata_df, columns_to_encode)

            # Apply encoding to each column
            metadata_df = self._apply_encodings(metadata_df, columns_to_encode, mappings)

            msg_info = "Function _encode_dataframe completed successfully"
            logger.info(msg_info, extra={"status": "success"})
            return metadata_df

        except Exception as e:
            logger.error(f"Error in function _encode_dataframe: {str(e)}", exc_info=True,
                         extra={"status": "failed", "error": str(e)})
            raise

    def _create_mappings(self, metadata_df: pd.DataFrame, 
                            columns_to_encode: Dict[str, str]) -> Dict[str, Dict]:
        """
        Creates mapping dictionaries for each categorical column.

        Args:
            metadata_df: The DataFrame containing metadata to be converted.
            columns_to_encode: Dictionary mapping column names to their mapping variable names.

        Returns:
            Dict[str, Dict]: A dictionary of mapping dictionaries for each column.
        """
        mappings = {}

        for column in columns_to_encode:
            values = metadata_df[column].unique().tolist()
            mappings[column] = self._create_string_to_int_mapper(values).mapping

        return mappings

    def _apply_encodings(self, metadata_df: pd.DataFrame,
                         columns_to_encode: List[str],
                         mappings: Dict[str, Dict]) -> pd.DataFrame:
        """
        Applies the encoding mappings to each specified column in the DataFrame.

        Args:
            metadata_df: The DataFrame containing metadata to be converted.
            columns_to_encode: Dictionary mapping column names to their mapping variable names.
            mappings: Dictionary of mapping dictionaries for each column.

        Returns:
            pd.DataFrame: The DataFrame with the specified columns converted to integer values.
        """
        for column in columns_to_encode:
            metadata_df[column] = metadata_df[column].map(mappings[column]).fillna(-1).astype(int)

        return metadata_df

    @log_method()
    def _create_string_to_int_mapper(self, strings: list, *,
                                        logger: Optional[logging.Logger] = None) -> callable:
        """Creates a mapping function between strings and integers.

        This is a factory function that generates a callable (the mapper)
        capable of converting predefined category strings into their corresponding
        integer indices (starting from 0).

        Args:
            strings (list): A list of strings to be mapped (e.g., ["Normal", "Stenosis"])
                            The order of the strings defines their integer value.

        Returns:
            callable: A function that maps an input string to an integer.
                      The resulting function includes 'mapping' and 'reverse_mapping
                      dictionaries as attributes.
        """

        logger = logger or self.logger
        logger.info("Starting function _create_string_to_int_mapper")

        try:
            # Create the primary dictionary {string: integer} using enumeration.
            # The integer (i) corresponds to the string's index in the list.
            mapping = {s: i for i, s in enumerate(strings)}

            # Create the optional reverse dictionary {integer: string}
            # for inspection/deserialization.
            reverse_mapping = {i: s for i, s in enumerate(strings)}

            def mapper(s: str) -> int:
                """Maps an input string to its corresponding integer index."""
                # Use .get() to return the mapped integer.
                # Returns -1 if the input string is not found (unknown category).
                return mapping.get(s, -1)

            # Attach the mapping dictionaries as attributes to the mapper function.
            # This allows users to inspect or reverse the mapping later.
            mapper.mapping = mapping
            mapper.reverse_mapping = reverse_mapping

            # Return the callable function
            logger.info("Function _create_string_to_int_mapper completed successfully",
                            extra={"status": "success"})
            return mapper

        except Exception as e:
            error_msg = f"Error in function _create_string_to_int_mapper : {str(e)}"
            logger.error(error_msg, exc_info=True, extra={"status": "failed", "error": str(e)})
