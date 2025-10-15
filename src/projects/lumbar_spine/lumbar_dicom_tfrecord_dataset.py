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
        self.logger = logger if logger is not None else logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing LumbarDicomTfRecordDataset")

        super().__init__(config, logger= self.logger) # Pass the logger to parent class
 

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


    @log_method()
    def _parse_tfrecord(self, example_proto: tf.Tensor,*,
                       logger: Optional[logging.Logger] = None) -> Tuple[tf.Tensor, Dict]:
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
        
            # Let's define a nested function to handle the output conversion within the Python context:
            def py_deserialize_and_flatten(metadata_bytes_tensor):
                # We MUST call .numpy() here, inside the py_function boundary.
                deserialized_dict = self._deserialize_metadata(metadata_bytes_tensor.numpy())
            
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
                flattened_records = [val for rec in records_list for val in rec] or [0.0] * 4 
            
                # Pad the flattened list to the MAX_RECORDS size (25 * 4 = 100 elements)
                MAX_RECORDS = 25
                MAX_RECORDS_FLAT = MAX_RECORDS * 4
            
                # Pad with a safe value (0.0)
                padded_records = flattened_records + [0.0] * (MAX_RECORDS_FLAT - len(flattened_records))
            
                records_tensor = tf.constant(padded_records, dtype=tf.float32)

                return header_tensors + [records_tensor]


            # Define the output signature for tf.py_function
            # 6 scalar headers (tf.int32) + 1 padded records tensor (tf.float32)
            output_types = [tf.int32] * 6 + [tf.float32]
        
            # Call the py_function
            deserialized_tensors = tf.py_function(
                py_deserialize_and_flatten, 
                inp=[metadata_bytes], 
                Tout=output_types
            )
        
            # Assign the results back to named Tensors
            (study_id_t, series_id_t, instance_number_t, description_t, condition_t, nb_records_t, padded_records_t) = deserialized_tensors
        
            # Reshape the padded records tensor to (MAX_RECORDS, 4)
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
                                   output_dir: str,*,
                                   logger: Optional[logging.Logger] = None) -> None:
        """
        Converts DICOM files stored in a hierarchical directory structure into 
        TensorFlow TFRecord files, generating one TFRecord file per study.

        This function iterates through all DICOM files, reads the image data, 
        retrieves the pre-serialized metadata, and writes both to a TFRecord file 
        for optimized data loading during training.

        Args:
            root_dir (str): The path to the root directory containing study subfolders 
                            (e.g., /data/dicom_root/).
            metadata_df (pd.DataFrame): A DataFrame containing pre-processed metadata 
                                        used to retrieve the serialized bytes.
            output_dir (str): The directory where the resulting TFRecord files will be saved.
            logger: Automatically injected logger (optional).

        Returns:
            None: The function saves files to disk but returns nothing.
        """
        
        logger = logger or self.logger
        logger.info("Starting DICOM to TFRecord conversion",
                   extra={"action": "convert_dicom", "root_dir": root_dir})

        try:
            # --- 1. Setup Output Directory ---
            # Initialize the output directory, creating it if it doesn't exist.
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Iterate over each study folder in the root directory, using tqdm for a progress bar.
            for study_path in tqdm(list(Path(root_dir).iterdir())):
                if not study_path.is_dir():
                    continue

                study_id = study_path.name
                # Define the output file path for the current study's TFRecord.
                output_path = output_dir / f"{study_id}.tfrecord"

                # Open the TFRecord writer, ensuring all records for the study go into one file.
                with tf.io.TFRecordWriter(str(output_path)) as writer:
                    # Iterate through series subfolders within the current study.
                    for series_path in study_path.iterdir():
                        if not series_path.is_dir():
                            continue

                        series_id = int(series_path.name)
                        # Iterate over all DICOM files (*.dcm) within the current series.
                        for dicom_path in series_path.glob("*.dcm"):
                    
                            # --- 2. Process Image Data ---
                    
                            # Load the DICOM image using SimpleITK.
                            img = sitk.ReadImage(str(dicom_path))
                            # Convert the SimpleITK image object to a NumPy array.
                            img_array = sitk.GetArrayFromImage(img)

                            # Convert the NumPy array to a TensorFlow Tensor, preserving the original type (e.g., uint16).
                            img_tensor = tf.convert_to_tensor(img_array, dtype=tf.uint16)
                            # Serialize the Tensor to a byte string for storage in the TFRecord.
                            img_bytes = tf.io.serialize_tensor(img_tensor).numpy()

                            # --- 3. Process Metadata ---
                    
                            # Call an assumed helper method to retrieve the pre-serialized metadata bytes 
                            # for the specific DICOM file from the main metadata DataFrame.
                            serialized_metadata = self._get_metadata_for_file(str(dicom_path), metadata_df)

                            # --- 4. Create and Write TFRecord Example ---
                    
                            # Create the feature dictionary structure required by tf.train.Example.
                            feature = {
                                "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_bytes])),
                                "metadata": tf.train.Feature(bytes_list=tf.train.BytesList(value=[serialized_metadata]))
                            }

                            # Assemble the features into a single Example protocol buffer.
                            example = tf.train.Example(features=tf.train.Features(feature=feature))
                    
                            # Serialize the Example and write it to the TFRecord file.
                            writer.write(example.SerializeToString())
        
            logger.info("DICOM to TFRecord conversion completed successfully",
                       extra={"status": "success"})
            
        except Exception as e:
            logger.error(f"Error during DICOM conversion: {str(e)}", exc_info=True,
                        extra={"status": "failed", "error": str(e)})
            raise


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

        The serialization format includes a fixed-size header followed by a variable 
        number of fixed-size records (the payload). The method ensures fixed-width 
        storage for all numerical data to optimize space and parsing consistency.

        Args:
            study_id (str/int): The study identifier used to filter the DataFrame.
            series_id (str/int): The series identifier used to filter the DataFrame.
            instance_number (str/int): The name of the DICOM file, without its '.dcm' extension,
                                        used to filter the DataFrame.
            data_df (pd.DataFrame): The main DataFrame containing all metadata records.
            logger: Automatically injected logger (optional).

        Returns:
            bytes: A compact byte sequence representing the serialized metadata.
               
                   Structure:
                   [Header (15 bytes)] + [Record 1 (8 bytes)] + ... + [Record N (8 bytes)]

        Raises:
            ValueError: If the number of records for the given identifiers exceeds 25.
        """
        
        logger = logger or self.logger
        logger.info("Starting function _serialize_metadata")

        try:

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
                metadata_bytes = b''
        
            else:
                # Extract global properties (must be unique for the filtered records)
                condition=int(records_df['condition'].unique()[0])
                description = int((records_df['series_description'].unique())[0])

                # --- Header Serialization (15 bytes total) ---

                # Serialize study_id and series_id as 5-byte unsigned integers (big-endian)
                study_id_bytes = int(study_id_str).to_bytes(5, byteorder='big', signed=False)
                series_id_bytes = int(series_id_str).to_bytes(5, byteorder='big', signed=False)

                # Serialize instance_number as a 2-byte unsigned short (max 9999)
                instance_number_bytes = struct.pack('=H', int(instance_number_str))

                # Serialize description and condition as 1-byte unsigned characters (sufficient for small values)
                description_bytes = struct.pack('=B', description) # '=B' → 1-byte formatting. Sufficient for 0, 1, 2
                condition_bytes = struct.pack('=B', condition) # '=B' → 1-byte formatting. Sufficient for 0, 1, 2, 3, 4

                # Serialize the number of records (max 25) as a 1-byte unsigned character
                nb_records = len(records_df)
                if nb_records > 25:
                    raise ValueError("The number of records exceeds the limit of 25.")
                nb_records_bytes = struct.pack('=B', nb_records)  # '=B' → 1-byte formatting. Sufficient for 0-255
 
                metadata_bytes = study_id_bytes + series_id_bytes + instance_number_bytes + description_bytes + condition_bytes + nb_records_bytes

                 # --- Payload Serialization (8 bytes per record) ---

                for row in records_df.itertuples():
                    level = row.level
                    severity = row.severity

                    # Convert float coordinates to integers by scaling (multiplication by 100) 
                    # to preserve two decimal places of precision
                    x = int(float(row.x) *100) # Conversion en int après multiplication par 100
                    y = int(float(row.y) *100)
            
                    # Serialize level and severity as 1-byte unsigned characters
                    level_bytes = struct.pack('=B', level)  # '=B' → 1-byte formatting. Sufficient for 0, 1, 2, 3, 4
                    severity_bytes = struct.pack('=B', severity)  # '=B' → 1-byte formatting. Sufficient for 0, 1, 2
            
                    # Serialize scaled x and y as 3-byte unsigned integers (big-endian)
                    x_bytes = int(x).to_bytes(3, byteorder='big', signed=False)
                    y_bytes = int(y).to_bytes(3, byteorder='big', signed=False)

                    metadata_bytes += level_bytes + severity_bytes + x_bytes + y_bytes
        
                logger.info("function _serialize_metadata completed successfully",
                                extra={"status": "success"})    
            return metadata_bytes

        except Exception as e:
            logger.error(f"Error in function _serialize_metadata() : {str(e)}", exc_info=True,
                        extra={"status": "failed", "error": str(e)})


    @log_method()
    def _deserialize_metadata(self, metadata_bytes: bytes,*,
                              logger: Optional[logging.Logger] = None) -> Dict:
        """
        Deserializes a compact byte sequence back into structured metadata components.

        This function is the inverse of _serialize_metadata. It parses the fixed-size 
        header and then reads a variable number of records based on the count found 
        in the header.

        Args:
            metadata_bytes (bytes): The byte sequence containing the serialized metadata.
            logger: Automatically injected logger (optional).

        Returns:
            dict: A dictionary containing the deserialized header values and a 
                  list of tuples (level, severity, x, y) for the individual records.
                  
                  Example:
                  {
                      'study_id': 12345, 
                      'series_id': 67890, 
                      'instance_number': 1, 
                      'description': 1, 
                      'condition': 3, 
                      'nb_records': 2, 
                      'records': [
                          (0, 1, 12.34, 56.78), # CHANGEMENT CLÉ: Tuple au lieu de Dict
                          (1, 0, 90.12, 34.56)
                      ]
                  }

        Raises:
            struct.error: If the byte sequence is shorter than expected (malformed data).
        """
        
        # NOTE: Suppression des logs à haute fréquence d'appel pour l'optimisation
        # logger = logger or self.logger
        # logger.info("Starting function _deserialize_metadata")

        # La gestion de l'erreur est déplacée dans le except final.
        try:
            if not metadata_bytes:
                # Retourne la structure minimale pour une séquence vide
                return {
                    'study_id': 0, 'series_id': 0, 'instance_number': 0, 
                    'description': 0, 'condition': 0, 'nb_records': 0, 
                    'records': [] # Liste de tuples vide
                }
            
            # Ensure the input is of type bytes
            if not isinstance(metadata_bytes, bytes):
                raise ValueError("Input must be a byte sequence.")
            
            # Check that the byte sequence is at least 15 bytes long (minimum header size)
            if len(metadata_bytes) < 15:
                raise struct.error("Byte sequence too short to contain valid metadata.")

            # --- Setup for Deserialization ---
            buffer = io.BytesIO(metadata_bytes)

            # --- Deserialize Header (15 bytes total) ---
            study_id = int.from_bytes(buffer.read(5), byteorder='big', signed=False)
            series_id = int.from_bytes(buffer.read(5), byteorder='big', signed=False)
            instance_number = struct.unpack('=H', buffer.read(2))[0]
            description = struct.unpack('=B', buffer.read(1))[0]
            condition = struct.unpack('=B', buffer.read(1))[0]
            nb_records = struct.unpack('=B', buffer.read(1))[0]

            # Initialize the results dictionary
            result = {
                'study_id': study_id,
                'series_id': series_id,
                'instance_number': instance_number,
                'description': description,
                'condition': condition,
                'nb_records': nb_records,
                'records': []
            }

            # --- Deserialize Payload (8 bytes per record) ---
            for _ in range(nb_records):
                # Read level (1 byte) and severity (1 byte)
                level = struct.unpack('=B', buffer.read(1))[0]
                severity = struct.unpack('=B', buffer.read(1))[0]
                
                # Read x (3 bytes)
                x_scaled = int.from_bytes(buffer.read(3), byteorder='big', signed=False)
                x = x_scaled / 100.0 # Rescale back to float
                
                # Read y (3 bytes)
                y_scaled = int.from_bytes(buffer.read(3), byteorder='big', signed=False)
                y = y_scaled / 100.0 # Rescale back to float

                # CHANGEMENT CLÉ: Append the deserialized record as a tuple
                result['records'].append((level, severity, x, y))

            # NOTE: Suppression du log de succès.
            return result

        except Exception as e:
            logger = logger or self.logger
            logger.error(f"Error in function _deserialize_metadata : {str(e)}", exc_info=True,
                        extra={"status": "failed", "error": str(e)})
            # En cas d'erreur de parsing, retourner une structure vide et sûre
            return {
                'study_id': 0, 'series_id': 0, 'instance_number': 0, 
                'description': 0, 'condition': 0, 'nb_records': 0, 
                'records': []
            }

    
    @log_method()
    def _encode_dataframe(self, metadata_df: pd.DataFrame,*,
                         logger: Optional[logging.Logger] = None) -> pd.DataFrame:
        """Converts categorical textual metadata fields in a DataFrame to numerical values.

        This process is essential for preparing data for serialization or machine 
        learning models, replacing descriptive strings with compact integers.

        Args:
            metadata_df (pd.DataFrame): The DataFrame containing metadata to be converted.
                                        Example column values: {"condition": "Spinal Canal Stenosis", "level": "L1-L2", ...}

        Returns:
            pd.DataFrame: The DataFrame with the specified columns converted to integer values.
        """
        
        logger = logger or self.logger
        logger.info("Starting function _encode_dataframe")

        try:
            # --- 1. Create Mapping Dictionaries ---
            # Generate unique integer mappings for each categorical column based on its unique values.
    
            # 1.1 Condition
            condition_values = metadata_df["condition"].unique().tolist()
            # Assume self._create_string_to_int_mapper returns an object with a 'mapping' attribute (dict).
            CONDITION_MAP = self._create_string_to_int_mapper(condition_values).mapping

            # 1.2 Level
            level_values = metadata_df["level"].unique().tolist()
            LEVEL_MAP = self._create_string_to_int_mapper(level_values).mapping

            # 1.3 Description
            description_values = metadata_df["series_description"].unique().tolist()
            DESCRIPTION_MAP = self._create_string_to_int_mapper(description_values).mapping

            # 1.4 Severity
            severity_values = metadata_df["severity"].unique().tolist()
            SEVERITY_MAP = self._create_string_to_int_mapper(severity_values).mapping

            # --- 2. Apply Encoding to the Columns ---

            # Use .map() to replace strings with their corresponding integer codes.
            # .fillna(-1) assigns a sentinel value (-1) to any string not found in the map (e.g., missing data).
            # .astype(int) converts the final result to the integer data type.
    
            metadata_df["condition"] = metadata_df["condition"].map(CONDITION_MAP).fillna(-1).astype(int)
            metadata_df["level"] = metadata_df["level"].map(LEVEL_MAP).fillna(-1).astype(int)
            metadata_df["series_description"] = metadata_df["series_description"].map(DESCRIPTION_MAP).fillna(-1).astype(int)
            metadata_df["severity"] = metadata_df["severity"].map(SEVERITY_MAP).fillna(-1).astype(int)

            logger.info("function _encode_dataframe completed successfully",
                            extra={"status": "success"}) 
            return metadata_df

        except Exception as e:
            logger.error(f"Error in function _encode_dataframe : {str(e)}", exc_info=True,
                        extra={"status": "failed", "error": str(e)})


    @log_method()
    def _create_string_to_int_mapper(self, strings: list,*,
                                    logger: Optional[logging.Logger] = None) -> callable:
        """Creates a mapping function between strings and integers.

        This is a factory function that generates a callable (the mapper) 
        capable of converting predefined category strings into their corresponding 
        integer indices (starting from 0).

        Args:
            strings (list): A list of strings to be mapped (e.g., ["Normal", "Stenosis"]). 
                            The order of the strings defines their integer value.

        Returns:
            callable: A function that maps an input string to an integer.
                      The resulting function includes 'mapping' and 'reverse_mapping' 
                      dictionaries as attributes.
        """

        logger = logger or self.logger
        logger.info("Starting function _create_string_to_int_mapper")
        
        try:
            # Create the primary dictionary {string: integer} using enumeration.
            # The integer (i) corresponds to the string's index in the list.
            mapping = {s: i for i, s in enumerate(strings)}

            # Create the optional reverse dictionary {integer: string} for inspection/deserialization.
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
            logger.error(f"Error in function _create_string_to_int_mapper : {str(e)}", exc_info=True,
                        extra={"status": "failed", "error": str(e)})
