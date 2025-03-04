from abc import ABC
from pathlib import Path

from ChestCancerClassifier.constants import *
from ChestCancerClassifier.utils.common import read_yaml, create_directories
from ChestCancerClassifier.entity.config_entity import *


class ConfigurationManagerTemplate(ABC):
    def __init__(
        self,
        config_filepath:  Path = CONFIG_FILE_PATH, # Path to configuration file - importet from constants
        params_filepath:  Path = PARAMS_FILE_PATH  # Path to parameters file - importet from constants
    ) -> None:
        
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)


class ConfigurationManager(ConfigurationManagerTemplate):
    """Manages project configurations, including data ingestion settings."""

    def __init__(self) -> None:
        super().__init__()
        """
        Initializes the ConfigurationManager.
        """

        #Check if artifacts_root exist inside self.config
        if 'artifacts_root' not in self.config:
          raise KeyError("'artifacts_root' key not found in the configuration file.")

        create_directories([self.config.artifacts_root]) #Use dictionary acces to avoid future errors.

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        Retrieves and constructs the data ingestion configuration.

        Returns:
            A DataIngestionConfig object.

        Raises:
            KeyError: If necessary keys are missing from the configuration.
        """

        #Error Handling - Check if 'data_ingestion' exist before accessing.
        if 'data_ingestion' not in self.config:
            raise KeyError("'data_ingestion' key not found in the configuration file.")

        config = self.config.data_ingestion

        # Error Handling: Check if necessary keys are present within 'data_ingestion'
        required_keys = ['root_dir', 'kaggle_source']
        for key in required_keys:
            if key not in config:
                raise KeyError(f"'{key}' key not found in 'data_ingestion' section of the configuration file.")

        #Type validation
        if not isinstance(config.root_dir, str):
             raise TypeError("'root_dir' in 'data_ingestion' must be a string.")
        if not isinstance(config.kaggle_source, str):
             raise TypeError("'kaggle_source' in 'data_ingestion' must be a string.")

        create_directories([config.root_dir])

        #Adding default parameters.
        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            kaggle_source=config.kaggle_source,
        )

        return data_ingestion_config

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:

        # Error Handling - Check if 'prepare_base_model' exist before accessing.
        if 'prepare_base_model' not in self.config:
            raise KeyError("'prepare_base_model' key not found in the configuration file.")

        config = self.config.prepare_base_model
        params = self.params

        # Error Handling: Check if necessary keys are present within 'prepare_base_model'
        required_keys = ["root_dir" ,"base_model_path", "updated_base_model_path"]
        for key in required_keys:
            if key not in config:
                raise KeyError(f"'{key}' key not found in 'prepare_base_model' section of the configuration file.")

        #Type validation
        if not isinstance(config.root_dir, str):
             raise TypeError("'root_dir' in 'prepare_base_model' must be a string.")
        if not isinstance(config.base_model_path, str):
             raise TypeError("'base_model_path' in 'prepare_base_model' must be a string.")
        if not isinstance(config.updated_base_model_path, str):
            raise TypeError("'updated_base_model_path' in 'prepare_base_model' must be a string.")

        create_directories([config.root_dir])

        #Adding default parameters.
        prepare_base_model_config = PrepareBaseModelConfig(
            # from config.yaml
            root_dir = Path(config.root_dir),
            base_model_path = Path(config.base_model_path),
            updated_base_model_path = Path(config.updated_base_model_path),
            # from params.yaml
            params_augmentation  = params.AUGMENTATION,
            params_image_size = params.IMAGE_SIZE,
            params_batch_size = params.BATCH_SIZE,
            params_include_top = params.INCLUDE_TOP,
            params_epochs = params.EPOCHS,
            params_classes = params.CLASSES,
            params_weights = params.WEIGHTS,
            params_learning_rate = params.LEARNING_RATE

        )

        return prepare_base_model_config

    def get_train_model_config(self) -> TrainModelConfig:
        # Error Handling - check if 'train_model' exist before accessing.
        if 'train_model' not in self.config:
            raise KeyError("'train_model' key not found in the configuration file.")

        config = self.config
        params = self.params

        # Error Handling: Check if necessary keys are present within 'train_model'
        required_keys = ["root_dir" ,"model_path"]
        for key in required_keys:
            if key not in config.train_model:
                raise KeyError(f"'{key}' key not found in 'train_model' section of the configuration file.")

        # Type validation
        if not isinstance(config.train_model.root_dir, str):
             raise TypeError("'root_dir' in 'train_model' must be a string.")
        if not isinstance(config.train_model.model_path, str):
             raise TypeError("'model_path' in 'train_model' must be a string.")

        create_directories([config.train_model.root_dir])

        #Adding default parameters.
        train_model_config = TrainModelConfig(
            # from config.yaml
            root_dir = Path(config.train_model.root_dir),
            model_path = Path(config.train_model.model_path),
            updated_base_model_path = Path(config.prepare_base_model.updated_base_model_path),
            ingested_data_path = Path(config.data_ingestion.root_dir),
            # from params.yaml
            params_augmentation  = params.AUGMENTATION,
            params_image_size = params.IMAGE_SIZE,
            params_batch_size = params.BATCH_SIZE,
            params_include_top = params.INCLUDE_TOP,
            params_epochs = params.EPOCHS,
            params_classes = params.CLASSES,
            params_weights = params.WEIGHTS,
            params_learning_rate = params.LEARNING_RATE

        )

        return train_model_config
