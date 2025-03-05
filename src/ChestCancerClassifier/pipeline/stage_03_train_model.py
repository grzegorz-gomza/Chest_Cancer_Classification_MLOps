# Standard library imports
from textwrap import dedent

# Local application/library imports
from ChestCancerClassifier.config.configuration import ConfigurationManager
from ChestCancerClassifier.components.train_model import TrainModel
from ChestCancerClassifier import logger
from ChestCancerClassifier.utils.common import start_stage_logger, end_stage_logger


STAGE_NAME = "Model Fine-Tuning"
class TrainModelPipeline:
    @staticmethod
    def main():
        # Readig YAML Files
        config = ConfigurationManager()
        
        # Create data ingestion configuration
        train_model_config = config.get_train_model_config() 

        # Download the database
        train_model = TrainModel(config=train_model_config)
        train_model.train_model()

if __name__ == "__main__":
    try:
        logger.info(start_stage_logger(STAGE_NAME))
        TrainModelPipeline.main()
        logger.info(end_stage_logger(STAGE_NAME))
    except Exception as e:
        logger.exception(e)
        raise e