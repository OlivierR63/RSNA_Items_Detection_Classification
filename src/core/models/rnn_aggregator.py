import tensorflow as tf
from tensorflow.keras import layers
from src.core.models.model_3d import BaseAggregator

class RNNAggregator(BaseAggregator):
    def build(self, x):
        x = self.resize_sequence(x)
        
        # Utilisation d'un LSTM bidirectionnel pour voir dans les deux sens de la colonne
        x = layers.Bidirectional(layers.LSTM(units=self.params.get('units', 256), 
                                             return_sequences=False))(x)
        return x