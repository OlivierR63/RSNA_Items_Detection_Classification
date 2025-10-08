# coding: utf-8

import pandas as pd
import tensorflow as tf
from typing import Tuple, Dict, Any

class CSVMetadata:
    """
    Handles the loading, merging, and preprocessing of metadata from three 
    separate CSV files (series descriptions, label coordinates, and training labels).

    The primary goal is to create a single, comprehensive DataFrame used later 
    for serialization and data lookup.
    """
    def __init__(self, series_description: str, label_coordinates: str, train: str) -> None:
        """
        Initializes the CSVMetadata object by loading the three required CSV files.

        Args:
            series_description (str): Path to the CSV file containing series descriptions.
            label_coordinates (str): Path to the CSV file containing label coordinates (x, y).
            train (str): Path to the CSV file containing training labels (severity levels).
        """
        # Load and ensure all data is treated as string initially to prevent merging errors.
        self._series_desc_df = pd.read_csv(series_description).astype(str)
        self._label_coords_df = pd.read_csv(label_coordinates).astype(str)
        self._train_df = pd.read_csv(train).astype(str)
        
        # Perform the initial merge and preprocessing upon instantiation.
        self._merged_df = self._merge_metadata()

    @property
    def merged(self) -> pd.DataFrame:
        """Returns the final, merged, and preprocessed DataFrame."""
        return self._merged_df

    @property
    def train_df(self) -> pd.DataFrame:
        """Returns the raw training label DataFrame."""
        return self._train_df
    
    def _merge_metadata(self) -> pd.DataFrame:
        """
        Merges the self._train_df successively with self._label_coords_df and 
        self._series_desc_df after first 'melting' (unpivoting) the train DataFrame.
        """
        # 1. Melt/Unpivot the train DataFrame
        # Convert columns representing different conditions/levels into rows.
        tmp_train_df = self.train_df.melt(
            id_vars="study_id", 
            var_name="condition_level", 
            value_name="Severity"
        )
        
        # Remove entries where no Severity is recorded.
        tmp_train_df.dropna(inplace=True)
        
        # 2. Extract 'condition' and 'level' from 'condition_level'
        # Example: 'L1_S1_level' needs to be split and cleaned.
        tmp_train_df[['condition', 'level']] = tmp_train_df['condition_level'].apply(
            lambda x: x[:-6] + ' ' + x[-5:]
        ).str.split(' ', expand=True)
        
        # Clean up the condition string (e.g., replace underscores with spaces).
        tmp_train_df['condition'] = tmp_train_df['condition'].str.replace('_',' ')
        # Clean up and normalize the level string (e.g., 'l1_s1' -> 'L1/S1').
        tmp_train_df['level'] = tmp_train_df['level'].str.replace('_','/').str.upper()

        # 3. Merge with Label Coordinates
        # Merge on three keys: study_id, condition, and level.
        tmp_train_df = tmp_train_df.merge(
            self._label_coords_df, 
            on=["study_id", 'condition', 'level'], 
            how='left'
        )
        # The temporary column is no longer needed.
        tmp_train_df.drop(['condition_level'], axis=1, inplace=True)
        
        # 4. Merge with Series Descriptions
        # Final merge to add series-specific metadata (like series_id).
        return tmp_train_df.merge(
            self._series_desc_df, 
            on=['study_id', 'series_id'], 
            how ='left'
        )

    def to_tf_lookup(self) -> tf.lookup.StaticHashTable:
        """
        Converts the merged metadata into a TensorFlow StaticHashTable for 
        efficient lookup within the TensorFlow graph.

        The key is a concatenated string of identifiers, and the value is 
        an array of key metadata fields.
        
        Returns:
            tf.lookup.StaticHashTable: A lookup table ready for use in a tf.data pipeline.
        """
        # Create unique keys: "study_id_series_id_instance_number"
        keys = self.merged["study_id"] + "_" + self.merged["series_id"] + "_" + self.merged["instance_number"].astype(str)
        
        # Create values: an array of [condition, level, x, y] strings.
        values = self.merged[["condition", "level", "x", "y"]].values.astype(str)
        
        # Create the hash table. Unknown keys will return an empty string.
        return tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys, values), 
            default_value="")
