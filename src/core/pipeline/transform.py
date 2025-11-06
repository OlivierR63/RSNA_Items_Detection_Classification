import tensorflow as tf
from tfx.components import Transform
from tf_transform.tf_metadata import schema_utils
from tfx.components.transform.component import Transform as TftTransform
from tfx.components.transform import component

@component
def Transform(
    examples: Input[standard_artifacts.Examples],
    schema: Input[standard_artifacts.Schema],
    module_file: str
) -> Output[standard_artifacts.TransformGraph]:

    """Composant Transform pour le prétraitement des features."""
    def preprocessing_fn(inputs):
        """Fonction de prétraitement."""
        outputs = {}

        # Normalisation de l'image
        outputs["image"] = tf.cast(inputs["image"], tf.float32) / 255.0

        # Encodage des features catégorielles
        outputs["condition"] = tf.one_hot(inputs["condition"], depth=5)
        outputs["level"] = tf.one_hot(inputs["level"], depth=5)
        outputs["severity"] = tf.one_hot(inputs["severity"], depth=3)

        # Coordonnées (x, y) normalisées
        outputs["x"] = (inputs["x"] - 500.0) / 500.0  # Normalise entre -1 et 1
        outputs["y"] = (inputs["y"] - 500.0) / 500.0

        return outputs

    # Charge le module de transformation
    transform_module = importlib.import_module(module_file)
    transform_fn = transform_module.preprocessing_fn

    # Applique la transformation
    transform = TftTransform(
        examples=examples,
        schema=schema,
        preprocessing_fn=preprocessing_fn
    )

    return transform.outputs['transform_graph']