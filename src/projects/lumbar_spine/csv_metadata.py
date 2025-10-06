# coding: utf-8

import pandas as pd
import tensorflow as tf

class CSVMetadata:
    def __init__(self, series_description: str, label_coordinates: str, train: str):
        self.series_desc = pd.read_csv(series_description).astype(str)
        self.label_coords = pd.read_csv(label_coordinates).astype(str)
        self.train = pd.read_csv(train).astype(str)
        self.merged = self._merge_metadata()

    def _merge_metadata(self) -> pd.DataFrame:
        tmp_merge_df = self.label_coords.merge(self.series_desc, on=["study_id", "series_id"], how='outer')
        tmp_merge_df['id'] = (tmp_merge_df['study_id'] + "_" + tmp_merge_df['condition'] + "_" + tmp_merge_df['level']).str.replace(' ', '_').str.replace('/', '_').str.lower()
        tmp_merge_df.drop(['study_id', 'condition', 'level'], axis=1, inplace=True)

        tmp_train_df = self.train.melt(id_vars="study_id", var_name="position_level", value_name="Severity")
        tmp_train_df['id'] = (tmp_train_df['study_id'] + "_" + tmp_train_df['position_level']).str.replace(' ', '_').str.lower()
        tmp_train_df.drop(['position_level'], axis=1, inplace=True)
        
        return tmp_train_df.merge(tmp_merge_df, on="id")

    def to_tf_lookup(self) -> tf.lookup.StaticHashTable:
        """Convertit les metadonnees en une table de lookup TensorFlow."""
        keys = self.merged["study_id"] + "_" + self.merged["series_id"] + "_" + self.merged["instance_number"].astype(str)
        values = self.merged[["condition", "level", "x", "y"]].values.astype(str)
        return tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys, values), default_value=""
        )
