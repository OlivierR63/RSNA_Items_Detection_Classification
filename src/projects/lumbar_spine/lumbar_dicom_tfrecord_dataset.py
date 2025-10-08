# coding: utf-8

# src/core/data_handlers/dicom_tfrecord_dataset.py
import tensorflow as tf
from typing import Dict, Tuple, List
from src.core.data_handlers.dicom_tfrecord_dataset import DicomTFRecordDataset
from src.projects.lumbar_spine.csv_metadata import CSVMetadata
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import SimpleITK as sitk
import struct
import io

class LumbarDicomTFRecordDataset(DicomTFRecordDataset):
    """TensorFlow Dataset for loading DICOM TFRecords."""


    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.generate_tfrecord_files()  # Generate the TFRecords if needed.
 

    def generate_tfrecord_files(self) -> None:
        """
        Generates TFRecord files from DICOM images and associated metadata.

        This function performs the following steps:
        1. Creates the necessary output directory for TFRecord files.
        2. Loads and merges metadata from CSV files as specified in the configuration.
        3. Encodes categorical metadata fields (e.g., condition, level, series_description, severity)
           into numerical values for compatibility with machine learning pipelines.
        4. Converts DICOM files to TFRecord format (if no existing TFRecord files are found in the output directory).

        Args:
            None (relies on instance attributes: `_tfrecord_pattern`, `_config`, `_tfrecord_dir`).

        Notes:
            - TFRecord files are only generated if the output directory is empty.
            - The DICOM root directory and output directory are specified in the configuration.
        """

        # 2. Prepare the directories
        self._tfrecord_pattern.mkdir(parents=True, exist_ok=True)

        # 3. Load and merge metadata
        metadata_handler = CSVMetadata(**self._config["csv_files"])

        # 4. Replace categorical values with numerical values
        #    Fields affected : condition, level, series_description, severity
        encoded_metadata_df = self.encode_dataframe(metadata_handler._merged_df) 

        # 4. Convert DICOM files to TFRecords (if needed)
        if not list(self._tfrecord_dir.glob("*.tfrecord")):
            print("Creating TFRecord files...")
            self.convert_dicom_to_tfrecords(
                root_dir=self._config["dicom_root_dir"],
                metadata_df=encoded_metadata_df,
                output_dir=str(self._tfrecord_dir)
            )
        print("TFRecords successfully created.")


    def _parse_tfrecord(self, example_proto: tf.Tensor) -> Tuple[tf.Tensor, Dict]:
        """
        Parses a single TFRecord entry, extracting and processing the image and metadata.

        This function is typically used within a tf.data pipeline to transform 
        raw TFRecord strings into usable tensors and structured data.

        Args:
            example_proto (tf.Tensor): A scalar string Tensor representing one 
                                       serialized tf.train.Example protocol buffer.

        Returns:
            Tuple[tf.Tensor, Dict]: A tuple containing the processed image tensor 
                                    and a dictionary of the deserialized metadata.
        """
        # Define the structure of the features stored in the TFRecord.
        # We expect two features, both stored as serialized strings.
        feature_description = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "metadata": tf.io.FixedLenFeature([], tf.string)
        }
    
        # Parse the scalar protocol buffer string into a dictionary of Tensors.
        example = tf.io.parse_single_example(example_proto, feature_description)

        # --- 1. Deserialize and Reshape the Image Tensor ---
    
        # The image is stored as a serialized Tensor string. Parse it back into a float32 Tensor.
        image = tf.io.parse_tensor(example["image"], out_type=tf.float32)
    
        # Reshape the 1D Tensor back into its original 4D shape (e.g., [Depth, Height, Width, Channels]).
        # The shape [64, 64, 64, 1] suggests a 3D medical image (like a volume) with one channel.
        image = tf.reshape(image, [64, 64, 64, 1])  

        # --- 2. Deserialize the Metadata ---
    
        # The metadata is stored as the compact byte sequence we designed earlier.
        metadata_bytes = example["metadata"]
    
        # IMPORTANT: Since self.deserialize_metadata is a standard Python function, 
        # we must call .numpy() on the Tensor to retrieve the raw bytes value 
        # for processing outside the TensorFlow graph context.
        study_id, series_id, instance_number, description, condition, nb_records, records = \
            self.deserialize_metadata(metadata_bytes.numpy())

        # --- 3. Normalize the Image ---
    
        # Normalize the image data (e.g., scaling pixel values to the range [0, 1]).
        # This division by the maximum value is a simple normalization technique.
        image = image / tf.reduce_max(image)

        # --- 4. Return Processed Data ---
    
        # Return the processed image tensor and the structured metadata dictionary.
        return image, {
            "study_id": study_id,
            "series_id": series_id,
            "description": description,
            "condition": condition,
            "nb_records": nb_records,
            # Assumes self.parse_records further processes the records list 
            # into a more usable format.
            "records": self.parse_records(nb_records, records)
        }


    def parse_records(nb_records: int, records: list) -> List[Dict]:
        """
        Converts a list of raw record tuples into a list of structured dictionaries.

        This function takes the raw output (list of tuples) from the metadata 
        deserialization and assigns meaningful keys to each element, making the 
        data easy to access and interpret.

        Args:
            nb_records (int): The expected number of records (although not strictly used 
                              for looping, it confirms the count).
            records (list): A list where each item is a tuple or list containing 
                            (level, severity, x, y) values.

        Returns:
            List[Dict]: A list of dictionaries, where each dictionary represents 
                        one structured record with keys "level", "severity", "x", 
                        and "y".
        """
        rec_list = []
    
        # Iterate through each raw record tuple in the input list.
        for rec in records:
            # Unpack the tuple into named variables for clarity.
            # Assumes the order is (level, severity, x, y) as defined during serialization.
            level, severity, x, y = rec
        
            rec_dict = {}
            # Assign meaningful keys to the deserialized values.
            rec_dict["level"] = level
            rec_dict["severity"] = severity
            rec_dict["x"] = x
            rec_dict["y"] = y
        
            rec_list.append(rec_dict)
        
        return rec_list


    def create_tf_dataset(self, batch_size: int = 8) -> tf.data.Dataset:
        """
        Creates an optimized TensorFlow Dataset for training or evaluation by reading 
        and processing TFRecord files asynchronously.

        The pipeline is constructed to maximize I/O and processing throughput 
        using interleave, shuffle, and prefetch mechanisms.

        Args:
            batch_size (int): The number of elements to combine into a single batch. 
                              Defaults to 8.

        Returns:
            tf.data.Dataset: An optimized Dataset where each element is a tuple 
                             (image_tensor, metadata_dict).
        """
        # 1. List all TFRecord files matching the pattern (e.g., 'data/*.tfrecord').
        # Shuffling the file names helps ensure better data mixing across epochs.
        tfrecord_files = tf.data.Dataset.list_files(self.tfrecord_pattern, shuffle=True)

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

        return dataset


    def convert_dicom_to_tfrecords(self, root_dir: str, metadata_df: pd.DataFrame, output_dir: str) -> None:
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

    Returns:
        None: The function saves files to disk but returns nothing.
    """
    
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
                    
                    # --- 1. Process Image Data ---
                    
                    # Load the DICOM image using SimpleITK.
                    img = sitk.ReadImage(str(dicom_path))
                    # Convert the SimpleITK image object to a NumPy array.
                    img_array = sitk.GetArrayFromImage(img)

                    # Convert the NumPy array to a TensorFlow Tensor, preserving the original type (e.g., uint16).
                    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.uint16)
                    # Serialize the Tensor to a byte string for storage in the TFRecord.
                    img_bytes = tf.io.serialize_tensor(img_tensor).numpy()

                    # --- 2. Process Metadata ---
                    
                    # Call an assumed helper method to retrieve the pre-serialized metadata bytes 
                    # for the specific DICOM file from the main metadata DataFrame.
                    serialized_metadata = self.get_metadata_for_file(str(dicom_path), metadata_df)

                    # --- 3. Create and Write TFRecord Example ---
                    
                    # Create the feature dictionary structure required by tf.train.Example.
                    feature = {
                        "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_bytes])),
                        "metadata": tf.train.Feature(bytes_list=tf.train.BytesList(value=[serialized_metadata]))
                    }

                    # Assemble the features into a single Example protocol buffer.
                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    
                    # Serialize the Example and write it to the TFRecord file.
                    writer.write(example.SerializeToString())


    def get_metadata_for_file(self, file_path: str, metadata_df: pd.DataFrame) -> dict:
        """
        Returns the serialized metadata extracted from a dataframe and associated with a given DICOM file.

        This method derives the necessary identifiers (study ID, series ID, instance number) 
        from the file path structure and uses these to query and serialize the relevant 
        records from the comprehensive metadata DataFrame.

        Args:
            file_path (str): The full path to the DICOM file (used to derive IDs).
            metadata_df (pd.DataFrame): The main DataFrame containing all metadata records.

        Returns:
            bytes: The compact byte sequence representing the serialized metadata 
                   for the specific image, or an empty dictionary if the input DataFrame is None.
        """
        if metadata_df is None:
            # NOTE: Returning bytes instead of a dict, as the caller (convert_dicom_to_tfrecords) 
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
        serialized_metadata = self.serialize_metadata(study_id, series_id, instance_number, metadata_df)

        # Return the byte sequence.
        return serialized_metadata


    def serialize_metadata(self, study_id, series_id, instance_number, data_df):
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

        Returns:
            bytes: A compact byte sequence representing the serialized metadata.
               
                   Structure:
                   [Header (15 bytes)] + [Record 1 (8 bytes)] + ... + [Record N (8 bytes)]

        Raises:
            ValueError: If the number of records for the given identifiers exceeds 25.
        """
        mask = (
                (data_df["study_id"] == int(study_id)) &
                (data_df["series_id"] == int(series_id)) &
                (data_df["instance_number"] == int(instance_number))
                )
        records_df = data_df[mask]

        # Extract global properties (must be unique for the filtered records)
        condition=int(records_df['Condition'].unique()[0])
        description = int((records_df['series_description'].unique())[0])

        # --- Header Serialization (15 bytes total) ---

        # Serialize study_id and series_id as 5-byte unsigned integers (big-endian)
        study_id_bytes = int(study_id).to_bytes(5, byteorder='big', signed=False)
        series_id_bytes = int(series_id).to_bytes(5, byteorder='big', signed=False)

        # Serialize instance_number as a 2-byte unsigned short (max 9999)
        instance_number_bytes = struct.pack('=H', int(instance_number))

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
            severity = row.Severity

            # Convert float coordinates to integers by scaling (multiplication by 100) 
            # to preserve two decimal places of precision
            x = int(row.x.astype(float) *100) # Conversion en int après multiplication par 100
            y = int(row.y.astype(float) *100)
            
            # Serialize level and severity as 1-byte unsigned characters
            level_bytes = struct.pack('=B', level)  # '=B' → 1-byte formatting. Sufficient for 0, 1, 2, 3, 4
            severity_bytes = struct.pack('=B', severity)  # '=B' → 1-byte formatting. Sufficient for 0, 1, 2
            
            # Serialize scaled x and y as 3-byte unsigned integers (big-endian)
            x_bytes = int(x).to_bytes(3, byteorder='big', signed=False)
            y_bytes = int(y).to_bytes(3, byteorder='big', signed=False)

            metadata_bytes += level_bytes + severity_bytes + x_bytes + y_bytes

        return metadata_bytes


    def deserialize_metadata(data):
        """
        Deserializes a compact byte sequence back into structured metadata components.

        This function is the inverse of serialize_metadata. It parses the fixed-size 
        header and then reads a variable number of records based on the count found 
        in the header.

        Args:
            metadata_bytes (bytes): The byte sequence containing the serialized metadata.

        Returns:
            dict: A dictionary containing the deserialized header values and a 
                  list of dictionaries for the individual records.
              
                  Example:
                  {
                      'study_id': 12345, 
                      'series_id': 67890, 
                      'instance_number': 1, 
                      'description': 1, 
                      'condition': 3, 
                      'nb_records': 2, 
                      'records': [
                          {'level': 0, 'severity': 1, 'x': 12.34, 'y': 56.78},
                          {'level': 1, 'severity': 0, 'x': 90.12, 'y': 34.56}
                      ]
                  }

        Raises:
            struct.error: If the byte sequence is shorter than expected (malformed data).
        """
        # Use io.BytesIO for efficient reading of the byte sequence
        buffer = io.BytesIO(metadata_bytes)
    
        # --- Deserialize Header (15 bytes total) ---
    
        # Read study_id (5 bytes, big-endian, unsigned)
        study_id_bytes = buffer.read(5)
        study_id = int.from_bytes(study_id_bytes, byteorder='big', signed=False)

        # Read series_id (5 bytes, big-endian, unsigned)
        series_id_bytes = buffer.read(5)
        series_id = int.from_bytes(series_id_bytes, byteorder='big', signed=False)

        # Read instance_number (2 bytes, unsigned short '=H')
        # Unpack returns a tuple, so we take the first element [0]
        instance_number = struct.unpack('=H', buffer.read(2))[0]

        # Read description (1 byte, unsigned char '=B')
        description = struct.unpack('=B', buffer.read(1))[0]
    
        # Read condition (1 byte, unsigned char '=B')
        condition = struct.unpack('=B', buffer.read(1))[0]
    
        # Read nb_records (1 byte, unsigned char '=B'). This determines the loop count.
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
            # Read level (1 byte, unsigned char '=B')
            level = struct.unpack('=B', buffer.read(1))[0]
        
            # Read severity (1 byte, unsigned char '=B')
            severity = struct.unpack('=B', buffer.read(1))[0]
        
            # Read x (3 bytes, big-endian, unsigned)
            x_bytes = buffer.read(3)
        
            # The stored integer represents 100 * actual float value.
            x_scaled = int.from_bytes(x_bytes, byteorder='big', signed=False)
            x = x_scaled / 100.0 # Rescale back to float
        
            # Read y (3 bytes, big-endian, unsigned)
            y_bytes = buffer.read(3)
            # The stored integer represents 100 * actual float value.
            y_scaled = int.from_bytes(y_bytes, byteorder='big', signed=False)
            y = y_scaled / 100.0 # Rescale back to float

            # Append the deserialized record
            result['records'].append({'level': level, 'severity': severity, 'x': x, 'y': y})

        return result


    
    def encode_dataframe(self, metadata_df: pd.DataFrame) -> pd.DataFrame:
        """Converts categorical textual metadata fields in a DataFrame to numerical values.

        This process is essential for preparing data for serialization or machine 
        learning models, replacing descriptive strings with compact integers.

        Args:
            metadata_df (pd.DataFrame): The DataFrame containing metadata to be converted.
                                        Example column values: {"condition": "Spinal Canal Stenosis", "level": "L1-L2", ...}

        Returns:
            pd.DataFrame: The DataFrame with the specified columns converted to integer values.
        """

        # --- 1. Create Mapping Dictionaries ---
        # Generate unique integer mappings for each categorical column based on its unique values.
    
        # 1.1 Condition
        condition_values = metadata_df["condition"].unique().tolist()
        # Assume self.create_string_to_int_mapper returns an object with a 'mapping' attribute (dict).
        CONDITION_MAP = self.create_string_to_int_mapper(condition_values).mapping

        # 1.2 Level
        level_values = metadata_df["level"].unique().tolist()
        LEVEL_MAP = self.create_string_to_int_mapper(level_values).mapping

        # 1.3 Description
        description_values = metadata_df["description"].unique().tolist()
        DESCRIPTION_MAP = self.create_string_to_int_mapper(description_values).mapping

        # 1.4 Severity
        severity_values = metadata_df["severity"].unique().tolist()
        SEVERITY_MAP = self.create_string_to_int_mapper(severity_values).mapping

        # --- 2. Apply Encoding to the Columns ---

        # Use .map() to replace strings with their corresponding integer codes.
        # .fillna(-1) assigns a sentinel value (-1) to any string not found in the map (e.g., missing data).
        # .astype(int) converts the final result to the integer data type.
    
        metadata_df["condition"] = metadata_df["condition"].map(CONDITION_MAP).fillna(-1).astype(int)
        metadata_df["level"] = metadata_df["level"].map(LEVEL_MAP).fillna(-1).astype(int)
        metadata_df["description"] = metadata_df["description"].map(DESCRIPTION_MAP).fillna(-1).astype(int)
        metadata_df["severity"] = metadata_df["severity"].map(SEVERITY_MAP).fillna(-1).astype(int)

        return metadata_df


    def create_string_to_int_mapper(self, strings: list) -> callable:
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
        return mapper
