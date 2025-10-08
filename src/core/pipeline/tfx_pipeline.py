from tfx.components import (
    ExampleGen, StatisticsGen, SchemaGen, Transform,
    Trainer, Tuner, Evaluator, Pusher, InfraValidator
)
from tfx.proto import trainer_pb2, pusher_pb2, infra_validator_pb2
from tfx.dsl.components.common import resolver
from tfx.dsl.inputs import ExternalInput
from tfx.types import Channel, standard_artifacts
from tfx.orchestration import pipeline

def create_tfx_pipeline(
                        data_root: str,
                        module_file: str,
                        serving_model_dir: str,
                        metadata_path: str
                        ) -> pipeline.Pipeline:
    """Crée un pipeline TFX complet."""
    # 1. Lecture des données
    example_gen = ExampleGen(
        input_base=data_root,
        output_config=standard_artifacts.Examples(
            split_config=standard_artifacts.SplitConfig(splits=[
                standard_artifacts.SplitConfig.Split(name='train', hash_buckets=8),
                standard_artifacts.SplitConfig.Split(name='eval', hash_buckets=2)
            ])
        )
    )

    # 2. Génération des statistiques et schéma
    statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])
    schema_gen = SchemaGen(statistics=statistics_gen.outputs['statistics'])

    # 3. Transformation des features
    transform = Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        module_file=module_file
    )

    # 4. Entraînement du modèle
    trainer = Trainer(
        module_file=module_file,
        examples=transform.outputs['transformed_examples'],
        schema=transform.outputs['schema'],
        train_args=trainer_pb2.TrainArgs(num_steps=10000),
        eval_args=trainer_pb2.EvalArgs(num_steps=5000)
    )

    # 5. Évaluation
    evaluator = Evaluator(
        examples=example_gen.outputs['examples'],
        model=trainer.outputs['model'],
        baseline_model=resolver.Resolver(
            strategy_class=latest_blessed_model_resolver.LatestBlessedModelResolver,
            model=Channel(type=standard_artifacts.Model),
            model_blessing=Channel(type=standard_artifacts.ModelBlessing)
        ).outputs['model'],
        eval_config=evaluator_pb2.EvalConfig(
            model_specs=[evaluator_pb2.ModelSpec(label_key='condition')],
            metrics_specs=[...]
        )
    )

    # 6. Déploiement
    pusher = Pusher(
        model=trainer.outputs['model'],
        model_push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=serving_model_dir
            )
        )
    )

    # 7. Validation de l'infrastructure
    infra_validator = InfraValidator(
        model=trainer.outputs['model'],
        serving_spec=infra_validator_pb2.ServingSpec(
            tensorflow_serving=infra_validator_pb2.TensorFlowServing(
                tags=["latest"]
            )
        )
    )

    return pipeline.Pipeline(
        pipeline_name="lumbar_spine_tfx_pipeline",
        pipeline_root=metadata_path,
        components=[
            example_gen, statistics_gen, schema_gen, transform,
            trainer, evaluator, pusher, infra_validator
        ],
        enable_cache=True,
        metadata_connection_config=...
    )