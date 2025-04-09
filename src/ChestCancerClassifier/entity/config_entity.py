from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    kaggle_source: str

@dataclass(frozen=True)
class PrepareBaseModelConfig:
    # from config.yaml
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    # from params.yaml
    params_augmentation :bool
    params_random_flip : str
    params_random_rotation : float
    params_random_zoom : float
    params_random_contrast : float
    params_random_brightness : float
    params_image_size: list
    params_batch_size: int
    params_include_top: bool
    params_epochs: int
    params_classes: int
    params_weights: str
    params_learning_rate: float
    params_use_tf_dataset: bool
    params_use_categorical_encoding: bool
    params_use_pretrained_model: bool

@dataclass(frozen=True)
class TrainModelConfig:
    # Params from config.yaml - Step 3
    root_dir: Path
    model_path: Path
    # Params from config.yaml - Step 2
    updated_base_model_path: Path
    # Params from config.yaml - Step 1
    ingested_data_path: Path
    # from params.yaml
    params_augmentation :bool
    params_image_size: list
    params_batch_size: int
    params_include_top: bool
    params_epochs: int
    params_classes: int
    params_weights: str
    params_learning_rate: float
    params_use_pretrained_model: bool
    params_use_pickled_data: bool
    params_use_tf_dataset: bool
    params_use_categorical_encoding: bool
    
@dataclass(frozen=True)
class EvaluationConfig:
    root_dir: Path
    model_path: Path
    model_path_deploy: Path
    pickle_data_path: Path
    mlflow_uri: str
    all_params: dict
    params_image_size: list
    params_batch_size: int
    params_use_tf_dataset: bool
    params_use_categorical_encoding: bool