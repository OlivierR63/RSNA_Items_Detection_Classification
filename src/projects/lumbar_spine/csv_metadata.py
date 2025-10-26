# coding: utf-8

import pandas as pd
import numpy as np
import tensorflow as tf
from typing import Optional
import logging

class CSVMetadata:
    """
    Handles the loading, merging, and preprocessing of metadata from three
    separate CSV files (series descriptions, label coordinates, and training labels).

    The primary goal is to create a single, comprehensive DataFrame used later 
    for serialization and data lookup.
    """
    def __init__(self,
                 series_description: str,
                 label_coordinates: str,
                 train: str,
                 logger: Optional[logging.Logger] = None) -> None:
        """
        Initializes the CSVMetadata object by loading the three required CSV files.

        Args:
            series_description (str): Path to the CSV file containing series descriptions.
            label_coordinates (str): Path to the CSV file containing label coordinates (x, y).
            train (str): Path to the CSV file containing training labels (severity levels).
            logger: Optional logger instance. If None, creates a new one.
        """
        self.logger = logger if logger is not None else logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing CSVMetadata handler",
                         extra={"action": "init",
                                "files": {
                                          "series_description": series_description,
                                          "label_coordinates": label_coordinates,
                                          "train": train
                                          }
                        })

        try:
            # Load and ensure all data is treated as string initially to prevent merging errors.
            self._series_desc_df = (
                                        pd.read_csv(series_description)
                                        .astype(str)
                                        .apply(lambda s: s.str.lower())
                                    )
            
            self._label_coords_df = (
                                        pd.read_csv(label_coordinates)
                                        .astype(str)
                                        .apply(lambda s: s.str.lower())
                                    )

            self._train_df = pd.read_csv(train).astype(str).apply(lambda s: s.str.lower())

            # Perform the initial merge and preprocessing upon instantiation.
            self._merged_df = self._merge_metadata()
            self.logger.info("Metadata merged successfully",
                             extra={"merged_shape": self._merged_df.shape})

        except Exception as e:
            self.logger.error(f"Error initializing CSVMetadata: {str(e)}",
                                 exc_info=True,
                                 extra={"status": "failed", "error": str(e)})
            raise

    @property
    def merged(self) -> pd.DataFrame:
        """Returns the final, merged, and preprocessed DataFrame."""
        self.logger.info("Accessing merged metadata", extra={"action": "get_merged"})
        return self._merged_df

    @property
    def train_df(self) -> pd.DataFrame:
        """Returns the raw training label DataFrame."""
        self.logger.info("Accessing raw training DataFrame", extra={"action": "get_train_df"})
        return self._train_df

    def _merge_metadata(self) -> pd.DataFrame:
        """
        Merges the training DataFrame with label coordinates and series descriptions.
        Includes detailed logging for each step of the merge process.
        """
        self.logger.info("Starting metadata merge process", extra={"action": "merge_metadata"})
        try:
            tmp_train_df = self._melt_and_clean_train_df()
            tmp_train_df = self._merge_with_label_coordinates(tmp_train_df)
            merged_df = self._merge_with_series_descriptions(tmp_train_df)
            merged_df = self._normalize_identifier_types(merged_df)
            self.logger.info("Metadata merge completed successfully",
                             extra={"status": "success", "final_shape": merged_df.shape})
            return merged_df
        except Exception as e:
            self.logger.error(f"Error merging metadata: {str(e)}",
                                 exc_info=True,
                                 extra={"status": "failed", "error": str(e)})
            raise

    def _melt_and_clean_train_df(self) -> pd.DataFrame:
        """Melt and clean the training DataFrame."""
        tmp_train_df = self._train_df.melt(
            id_vars="study_id",
            var_name="condition_level",
            value_name="severity"
        )
        initial_count = len(tmp_train_df)
        tmp_train_df.dropna(inplace=True)
        final_count = len(tmp_train_df)
        self.logger.info(f"Dropped {initial_count - final_count} NaN rows",
                         extra={
                                "step": 1,
                                "initial_count": initial_count,
                                "final_count": final_count
                                }
                            )

        tmp_train_df[['condition', 'level']] = tmp_train_df['condition_level'].apply(
            lambda x: x[:-6] + ' ' + x[-5:]
        ).str.split(' ', expand=True)

        tmp_train_df['condition'] = tmp_train_df['condition'].str.replace('_', ' ')
        tmp_train_df['level'] = tmp_train_df['level'].str.replace('_', '/')
        tmp_train_df.drop(['condition_level'], axis=1, inplace=True)
        return tmp_train_df

    def _merge_with_label_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Merge the DataFrame with label coordinates."""
        self.logger.info("Merging with label coordinates...", extra={"step": 3})
        merged_df = df.merge(
            self._label_coords_df,
            on=["study_id", 'condition', 'level'],
            how='inner'
        )
        self.logger.info(f"Merged with label coordinates. Shape: {merged_df.shape}",
                         extra={"step": 3, "shape": merged_df.shape})
        return merged_df

    def _merge_with_series_descriptions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Merge the DataFrame with series descriptions."""
        self.logger.info("Merging with series descriptions...", extra={"step": 4})
        merged_df = df.merge(self._series_desc_df, on=['study_id', 'series_id'], how='inner')
        self.logger.info(f"Merged with series descriptions. Final shape: {merged_df.shape}",
                         extra={"step": 4, "final_shape": merged_df.shape})
        return merged_df

    def _normalize_identifier_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize identifier types."""
        self.logger.info("Normalizing identifier types...", extra={"step": 5})
        df['study_id'] = df['study_id'].astype(np.int64)
        df['series_id'] = df['series_id'].astype(np.int64)
        df['instance_number'] = df['instance_number'].astype(int)
        df.columns = [c.lower() for c in df.columns]
        return df

    def to_tf_lookup(self) -> tf.lookup.StaticHashTable:
        """
        Converts the merged metadata into a TensorFlow StaticHashTable for
        efficient lookup within the TensorFlow graph.

        The key is a concatenated string of identifiers, and the value is
        an array of key metadata fields.

        Returns:
            tf.lookup.StaticHashTable: A lookup table ready for use in a tf.data pipeline.
        """

        self.logger.info("Creating TensorFlow lookup table", extra={"action": "create_lookup"})

        try:
            # Create unique keys: "study_id_series_id_instance_number"
            keys = (self.merged["study_id"] + "_" + self.merged["series_id"]
                        + "_" + self.merged["instance_number"].astype(str))

            # Create values: an array of [condition, level, x, y] strings.
            values = self.merged[["condition", "level", "x", "y"]].values.astype(str)

            # Create the hash table. Unknown keys will return an empty string.
            lookup_table = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(keys, values),
                default_value="")
            self.logger.info("TensorFlow lookup table created successfully",
                             extra={"status": "success", "num_entries": len(keys)})
            return lookup_table

        except Exception as e:
            self.logger.error(f"Error creating lookup table: {str(e)}",
                              exc_info=True,
                              extra={"status": "failed", "error": str(e)})
            raise
