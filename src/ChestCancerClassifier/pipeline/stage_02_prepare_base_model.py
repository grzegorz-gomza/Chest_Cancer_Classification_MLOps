# Standard library imports
from textwrap import dedent

# Local application/library imports
from ChestCancerClassifier.config.configuration import ConfigurationManager
from ChestCancerClassifier.components.prepare_base_model import PrepareBaseModel
from ChestCancerClassifier import logger
from ChestCancerClassifier.utils.common import start_stage_logger, end_stage_logger


STAGE_NAME = "Prepare Base Model stage"
class PrepareBaseModelPipeline:
    @staticmethod
    def main():
        # Readig YAML Files
        config = ConfigurationManager()
        
        # Create data ingestion configuration
        base_model_prep_config = config.get_prepare_base_model_config() 

        # Download the database
        base_model_prep = PrepareBaseModel(config=base_model_prep_config)
        base_model_prep.download_base_model()
        base_model_prep.update_base_model()

if __name__ == "__main__":
    try:
        logger.info(start_stage_logger(STAGE_NAME))
        PrepareBaseModelPipeline.main()
        logger.info(end_stage_logger(STAGE_NAME))
    except Exception as e:
        logger.exception(e)
        raise e