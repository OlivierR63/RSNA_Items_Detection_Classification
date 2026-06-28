# coding: utf-8

class OneOf:
    """
    Marker for a scalar value allowing multiple authorized types.
    """
    def __init__(self, *types):
        self.types = types


class Sequence:
    """
    Marker for a fixed-structure sequence.
    """
    def __init__(self, *structure):
        self.structure = structure


REQUIRED_SCHEMA = {
    "paths": {
        "dicom_studies": str,
        "tfrecord": {
            "read_only_dir": str,
            "read_write_dir": str
        },
        "tfrecord_metadata_cache": OneOf(str, dict),
        "output": str,
        "checkpoint": str,
        "log_mirror": str,
        "tf_cache": str,
        "inspection": str,
        "csv": {
            "series_description": str,
            "label_coordinates": str,
            "instances_series_format": str,
            "train": str
        }
    },
    "data_specs": {
        "series_depth_percentile": OneOf(int, float),
        "max_records_per_frame": int,
        "dataset_buffer_size_mb": int
        },
    "models": {
        "backbone_2d": {
            "type": str,
            "img_shape": Sequence(int, int, int),
            "freeze": bool,
            "scaling": {
                "min": OneOf(int, float),
                "max": OneOf(int, float)
            }
        },
        "head_3d": {
            "type": str,
            "filters": int
        }
    },
    "training": {
        "batch_size": int,
        "epochs": int,
        "train_split_ratio": OneOf(int, float),
        "loss_balancer": {
            "momentum": float,
            "min_weight": OneOf(int, float),
            "max_weight": OneOf(int, float)
        }
    },
    "optimizer": {
        "type": str,
        "learning_rate": float,
        "clipnorm": OneOf(int, float)
    },
    "callbacks": {
        "patience": int,
        "resume_mode": {"best", "last"}
    },
    "compilation": {
        "loss_weights": {
            "severity_output": OneOf(int, float),
            "location_output": OneOf(int, float)
        },
        "class_weights": {
            "Normal/Mild": OneOf(int, float),
            "Moderate": OneOf(int, float),
            "Severe": OneOf(int, float)
        },
        "run_eagerly": bool
    },
    "system": {
        "nb_cores": int,
        "log_retention_days": int,
        "seed": int,
        "memory_threshold_percent": int,
        "chunksize": int
    },
    "dataset_steering": {
        "interleave": {
            "parallel_files": int,
            "block_per_file": int,
            "deterministic": bool
        },
        "group_studies": int,
        "prefetch_batches": int,
        "num_parallel_calls": int,
        "use_cache": bool
    },
    "logging": {
        "level": {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"},
        "console_display": bool,
        "use_json": bool
    }
}
