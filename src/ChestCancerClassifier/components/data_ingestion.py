# Standard library imports
import os
from pathlib import Path

# Third-party library imports
import dotenv
dotenv.load_dotenv() # has to be executed before kaggle

import kaggle

# Local application/library imports
from ChestCancerClassifier import logger
from ChestCancerClassifier.utils.common import get_size
from ChestCancerClassifier.entity.config_entity import DataIngestionConfig

class DataIngestion:
    """
    Handles the downloading of datasets for the Chest Cancer Classifier project,
    including validations, logging, and data size calculations.
    """

    def __init__(self, config: DataIngestionConfig):
        """
        Initialization with configuration provided.

        :param config: Configuration object for data ingestion.
        """
        self.config = config
        

    def is_dataset_downloaded(self) -> bool:
        """
        Check if the dataset has already been downloaded.

        :return: True if dataset exists and is non-empty, False otherwise.
        """
        if os.path.exists(self.config.root_dir) and len(os.listdir(self.config.root_dir)) > 0:
            logger.info("Dataset is already downloaded.")
            return True
        else:
            logger.info("Dataset is not downloaded. Proceeding to download.")
            return False

    def validate_kaggle_credentials(self) -> None:
        """
        Validate Kaggle API credentials. Logs an error and raises an exception if credentials are missing.
        """

        kaggle_username = os.getenv("KAGGLE_USERNAME")
        kaggle_key = os.getenv("KAGGLE_KEY")

        if not kaggle_username or not kaggle_key:
            error_message = """
            Kaggle API credentials not found. Ensure the following environment variables are set:
              - KAGGLE_USERNAME
              - KAGGLE_KEY
            Add your credentials to a `.env` file or export them as environment variables.
            """
            logger.error(error_message)
            raise EnvironmentError(error_message)

        logger.info("Kaggle API credentials imported successfully.")

    def validate_config(self) -> None:
        """
        Validate the essential configurations from the config file.
        """
        if not self.config.kaggle_source:
            error_message = "The 'kaggle_source' variable is missing in the configuration file."
            logger.error(error_message)
            raise ValueError(error_message)

        if not os.path.exists(self.config.root_dir):
            logger.info(f"Root directory '{self.config.root_dir}' does not exist. Creating it.")
            os.makedirs(self.config.root_dir, exist_ok=True)

    def download_file(self) -> None:
        """
        Handles dataset download from Kaggle and ensures it is available locally.
        """
        logger.info("Starting dataset download process...")
        self.validate_config()

        # Check if dataset is already downloaded
        if self.is_dataset_downloaded():
            dataset_size = get_size(Path(self.config.root_dir))
            logger.info(f"Dataset already downloaded. Size: {dataset_size} kB")
            return

        # Validate Kaggle API credentials
        self.validate_kaggle_credentials()

        # Attempt to download the dataset
        try:
            logger.info(f"Downloading dataset from Kaggle: {self.config.kaggle_source}")
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(
                dataset=self.config.kaggle_source,
                path=self.config.root_dir,
                unzip=True
            )
            logger.info(f"Dataset '{self.config.kaggle_source}' has been successfully downloaded to '{self.config.root_dir}'.")
        except Exception as e:
            logger.error(f"Error occurred while downloading the dataset: {e}")
            raise

        # Verify download
        if self.is_dataset_downloaded():
            dataset_size = get_size(Path(self.config.root_dir))
            logger.info(f"Dataset download complete. Size: {dataset_size} kB.")
        else:
            error_message = f"Dataset download failed. The directory '{self.config.root_dir}' is empty."
            logger.error(error_message)
            raise RuntimeError(error_message)

    def rename_subfolders(self) -> None:
        """
        Rename subfolders in all directories based on the names in the first directory.
        
        Args:
            root_dir: The root directory containing folders to process
        """
        for root, dirs, _ in os.walk(self.config.root_dir):
            if any(folder in dirs for folder in ["train", "test", "valid"]):
                reference_folder_names = None
                
                for i, folder in enumerate(dirs):
                    folder_path = os.path.join(root, folder)
                    
                    # Get subfolder names from the first directory as reference
                    if i == 0:
                        reference_folder_names = os.listdir(folder_path)
                    # Rename subfolders in subsequent directories
                    else:
                        target_subfolder_names = os.listdir(folder_path)
                        
                        # Make sure we have the same number of subfolders
                        if len(reference_folder_names) != len(target_subfolder_names):
                            logger.info(f"Warning: Number of subfolders doesn't match in {folder_path}")
                            continue
                        
                        # Pair the source names with current names and rename
                        for ref_name, current_name in zip(reference_folder_names, target_subfolder_names):
                            current_path = os.path.join(folder_path, current_name)
                            new_path = os.path.join(folder_path, ref_name)
                            
                            # Avoid renaming if the name is already correct
                            if current_name != ref_name:
                                try:
                                    os.rename(current_path, new_path)
                                    logger.info(f"Renamed: {current_path} -> {new_path}")
                                except Exception as e:
                                    logger.info(f"Error renaming {current_path}: {e}")
                
                # Once we've processed one set of directories, break to avoid processing nested directories
                break
