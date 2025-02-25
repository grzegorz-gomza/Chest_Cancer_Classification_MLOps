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
    params_image_size: list
    params_batch_size: int
    params_include_top: bool
    params_epochs: int
    params_classes: int
    params_weights: str
    params_learning_rate: float

