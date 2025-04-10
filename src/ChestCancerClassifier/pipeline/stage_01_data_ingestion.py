# Standard library imports
from textwrap import dedent

from tensorflow.test import is_built_with_cuda
from tensorflow.config import list_physical_devices

# Local application/library imports
from ChestCancerClassifier.config.configuration import ConfigurationManager
from ChestCancerClassifier.components.data_ingestion import DataIngestion
from ChestCancerClassifier import logger
from ChestCancerClassifier.utils.common import start_stage_logger, end_stage_logger


STAGE_NAME = "Data Ingestion stage"
class DataIngestionTrainingPipeline:
    @staticmethod
    def main():
        # Readig YAML Files
        config = ConfigurationManager()
        
        # Create data ingestion configuration
        data_ingestion_config = config.get_data_ingestion_config() 

        # Download the database
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.rename_subfolders()

if __name__ == "__main__":
    try:
        logger.info(start_stage_logger(STAGE_NAME))
        logger.info("Num GPUs Available: ", len(list_physical_devices('GPU')))
        logger.info(list_physical_devices('GPU')) #show detail GPUs
        logger.info(is_built_with_cuda())
        DataIngestionTrainingPipeline.main()
        logger.info(end_stage_logger(STAGE_NAME))
    except Exception as e:
        logger.exception(e)
        raise e