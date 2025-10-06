import tfx
from tfx.components import CsvExampleGen, Trainer, Pusher
from tfx.proto import pusher_pb2
from src.core.models.model_factory import ModelFactory

def create_pipeline(config: dict, output_dir: str) -> tfx.dsl.Pipeline:
    """Crée un pipeline TFX pour l'entraînement et le déploiement."""
    example_gen = CsvExampleGen(input_base=config["data"]["csv"]["root_dir"])

    trainer = Trainer(
        module_file=config["trainer"]["module_file"],
        examples=example_gen.outputs["examples"],
        train_args=tfx.proto.TrainArgs(num_steps=config["trainer"]["num_steps"]),
        eval_args=tfx.proto.EvalArgs(num_steps=config["trainer"]["eval_steps"])
    )

    pusher = Pusher(
        model=trainer.outputs["model"],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=output_dir
            )
        )
    )

    return tfx.dsl.Pipeline(
        components=[example_gen, trainer, pusher],
        enable_cache=True,
        metadata_connection_config=tfx.orchestration.metadata.sqlite_metadata_connection_config(
            f"{output_dir}/metadata.db"
        )
    )
