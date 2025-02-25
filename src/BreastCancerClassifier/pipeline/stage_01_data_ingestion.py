# Standard library imports
from textwrap import dedent

# Local application/library imports
from BreastCancerClassifier.config.configuration import ConfigurationManager
from BreastCancerClassifier.components.data_ingestion import DataIngestion
from BreastCancerClassifier import logger
from BreastCancerClassifier.utils.common import start_stage_logger, end_stage_logger


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

if __name__ == "__main__":
    try:
        logger.info(start_stage_logger(STAGE_NAME))
        DataIngestionTrainingPipeline.main()
        logger.info(end_stage_logger(STAGE_NAME))
    except Exception as e:
        logger.exception(e)
        raise e