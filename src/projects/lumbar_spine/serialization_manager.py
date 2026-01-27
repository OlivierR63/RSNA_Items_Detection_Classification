# coding: utf-8

from ast import Try
import pandas as pd
import io
from typing import Dict, Tuple, List, Optional
import logging
from src.core.utils.logger import log_method
from src.config.config_loader import ConfigLoader
import inspect

class SerializationManager:
    def __init__(self, config, logger):
        self._config = config
        self._logger = logger

    @staticmethod
    def serialize_metadata(
        study_id: int,
        series_id: int,
        series_min: int,
        series_max: int,
        instance_id: int,
        img_height: int,
        img_width: int,
        description: int,
        pathologies_df: pd.DataFrame,
        max_records: int = 25
    ) -> bytes:
        """
        Serializes metadata records (header + payload) into a structured binary format.
        
        This static method is designed for high-performance preprocessing. It avoids 
        external dependencies and instance logging to remain compatible with 
        multithreaded environments.

        Binary Structure:
        - Header (22 bytes): study(5), series(5), min(2), max(2), inst(2), h(2), w(2), desc(1), count(1)
        - Payload (8 bytes * N): level(1), severity(1), x(3), y(3)

        Args:
            study_id, series_id: Unique identifiers.
            series_min, series_max: Pixel value range (signed).
            instance_id, img_height, img_width: DICOM instance properties.
            description: MRI orientation code.
            pathologies_df: DataFrame containing columns ['condition_level', 'severity', 'x', 'y'].
            max_records: Maximum number of records allowed (default 25).

        Returns:
            bytes: The complete serialized metadata block.

        Raises:
            ValueError: If null values are found or if record count exceeds max_records.
            RuntimeError: If binary conversion fails.
        """
        # --- Validation ---
        if pathologies_df is None or pathologies_df.empty:
            nb_records = 0
        else:
            # Check for nulls in critical columns only to save time
            critical_cols = ['condition_level', 'severity', 'x', 'y']
            if pathologies_df[critical_cols].isnull().any().any():
                raise ValueError("Null values detected in critical pathology columns.")
            
            nb_records = len(pathologies_df)

        if nb_records > max_records:
            raise ValueError(f"Record count {nb_records} exceeds limit of {max_records}.")

        try:
            # --- Header Construction (22 bytes) ---
            header = (
                study_id.to_bytes(5, byteorder='big') +
                series_id.to_bytes(5, byteorder='big') +
                series_min.to_bytes(2, byteorder='big', signed=True) +
                series_max.to_bytes(2, byteorder='big', signed=True) +
                instance_id.to_bytes(2, byteorder='big') +
                img_height.to_bytes(2, byteorder='big') +
                img_width.to_bytes(2, byteorder='big') +
                description.to_bytes(1, byteorder='big') +
                nb_records.to_bytes(1, byteorder='big')
            )

            # --- Payload Construction (8 bytes per record) ---
            payload_segments = []
            if nb_records > 0:
                for row in pathologies_df.itertuples():
                    # level(1), severity(1)
                    seg = int(row.condition_level).to_bytes(1, 'big')
                    seg += int(row.severity).to_bytes(1, 'big')
                    
                    # Coordinates: scaled by 100 and stored in 3 bytes
                    x_scaled = round(float(row.x) * 100)
                    y_scaled = round(float(row.y) * 100)
                    
                    seg += x_scaled.to_bytes(3, 'big')
                    seg += y_scaled.to_bytes(3, 'big')
                    
                    payload_segments.append(seg)

            return header + b''.join(payload_segments)

        except Exception as e:
            raise RuntimeError(f"Serialization failed: {str(e)}")

    @staticmethod
    def deserialize_metadata(metadata_blob: bytes) -> Dict:
        """
        Deserializes a compact byte sequence back into structured metadata components.
        
        This static method is optimized for high-frequency calls during model training.
        It bypasses instance-level logging and configuration to avoid overhead and 
        serialization issues within TensorFlow's data pipeline.

        The structure of the input bytes object is:
        [Header (22 bytes)] + [Payload (8 bytes * N records)]

        Args:
            metadata_blob (bytes): The binary data containing serialized metadata.

        Returns:
            Dict: A dictionary containing:
                - study_id (int)
                - series_id (int)
                - series_min (int)
                - series_max (int)
                - instance_number (int)
                - img_height (int)
                - img_width (int)
                - description (int)
                - nb_records (int)
                - records (List[Tuple]): List of (condition_level, severity, x, y)

        Raises:
            ValueError: If the input blob is empty or None.
            RuntimeError: If binary parsing fails due to corrupted or truncated data.
        """
        if not metadata_blob:
            raise ValueError("Input metadata_blob is empty or None.")

        try:
            # Use io.BytesIO for efficient sequential binary reading
            buffer = io.BytesIO(metadata_blob)

            # --- Header Deserialization (Fixed 22 bytes) ---
            # study_id (5 bytes), series_id (5 bytes)
            study_id = int.from_bytes(buffer.read(5), byteorder='big', signed=False)
            series_id = int.from_bytes(buffer.read(5), byteorder='big', signed=False)
            
            # series_min/max (2 bytes each, signed for DICOM pixel values)
            series_min = int.from_bytes(buffer.read(2), byteorder='big', signed=True)
            series_max = int.from_bytes(buffer.read(2), byteorder='big', signed=True)
            
            # instance/dimensions (2 bytes each)
            instance_number = int.from_bytes(buffer.read(2), byteorder='big', signed=False)
            img_height = int.from_bytes(buffer.read(2), byteorder='big', signed=False)
            img_width = int.from_bytes(buffer.read(2), byteorder='big', signed=False)
            
            # orientation and record count (1 byte each)
            description = int.from_bytes(buffer.read(1), byteorder='big', signed=False)
            nb_records = int.from_bytes(buffer.read(1), byteorder='big', signed=False)

            header_data = {
                'study_id': study_id,
                'series_id': series_id,
                'series_min_pixel_value': series_min,
                'series_max_pixel_value': series_max,
                'instance_number': instance_number,
                'img_height': img_height,
                'img_width': img_width,
                'description': description,
                'nb_records': nb_records
            }

            # --- Payload Deserialization (8 bytes per record) ---
            records = []
            for _ in range(nb_records):
                # level (1 byte), severity (1 byte)
                condition_level = int.from_bytes(buffer.read(1), byteorder='big', signed=False)
                severity = int.from_bytes(buffer.read(1), byteorder='big', signed=False)

                # x and y coordinates (3 bytes each, scaled by 100)
                x_scaled = int.from_bytes(buffer.read(3), byteorder='big', signed=False)
                y_scaled = int.from_bytes(buffer.read(3), byteorder='big', signed=False)
                
                # Rescale back to original float values
                records.append((condition_level, severity, x_scaled / 100.0, y_scaled / 100.0))

            header_data['records'] = records
            return header_data

        except Exception as e:
            # Re-raise as RuntimeError to be caught by the TF data pipeline
            raise RuntimeError(f"Failed to deserialize metadata: {str(e)}")
