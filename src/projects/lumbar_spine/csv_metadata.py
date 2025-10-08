# coding: utf-8

import pandas as pd
import tensorflow as tf

class CSVMetadata:
    def __init__(self, series_description: str, label_coordinates: str, train: str) -> None:
        self._series_desc_df = pd.read_csv(series_description).astype(str)
        self._label_coords_df = pd.read_csv(label_coordinates).astype(str)
        self._train_df = pd.read_csv(train).astype(str)
        self._merged_df = self._merge_metadata()


    def _merge_metadata(self) -> pd.DataFrame:
        '''Fusionne le dataframe self._train_df successivemet avec self._label_coords_df et self._series_desc_df
            après l'avoir mis à plat.'''

        tmp_train_df = self.train_df.melt(id_vars="study_id", var_name="condition_level", value_name="Severity")
        tmp_train_df.dropna(inplace=True)
        tmp_train_df[['condition', 'level']] = tmp_train_df['condition_level'].apply(lambda x:x[:-6] + ' ' + x[-5:]).str.split(' ', expand=True)
        tmp_train_df['condition'] = tmp_train_df['condition'].str.replace('_',' ')
        tmp_train_df['level'] = tmp_train_df['level'].str.replace('_','/').str.upper()

        tmp_train_df = tmp_train_df.merge(self._label_coords_df, on=["study_id", 'condition', 'level'], how='left')
        tmp_train_df.drop(['condition_level'], axis=1, inplace=True)
        
        return tmp_train_df.merge(self._series_desc_df, on=['study_id', 'series_id'], how ='left')


    def to_tf_lookup(self) -> tf.lookup.StaticHashTable:
        """Convertit les metadonnees en une table de lookup TensorFlow."""
        keys = self.merged["study_id"] + "_" + self.merged["series_id"] + "_" + self.merged["instance_number"].astype(str)
        values = self.merged[["condition", "level", "x", "y"]].values.astype(str)
        return tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys, values), default_value=""
        )
