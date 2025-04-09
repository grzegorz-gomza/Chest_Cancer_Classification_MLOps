# Local application/library imports
from ChestCancerClassifier.config.configuration import ConfigurationManager
from ChestCancerClassifier.components.evaluate_model import EvaluateModel
from ChestCancerClassifier import logger
from ChestCancerClassifier.utils.common import start_stage_logger, end_stage_logger


STAGE_NAME = "Evaluate Model"
class EvaluateModelPipeline:
    @staticmethod
    def main():
        # Readig YAML Files
        config = ConfigurationManager()
        
        # Create data ingestion configuration
        evaluate_model_config = config.get_evaluation_config() 

        # Download the database
        evaluate_model = EvaluateModel(config=evaluate_model_config)
        evaluate_model.evaluate()
        evaluate_model.save_score()
        evaluate_model.mlflow_tracking() # comment out for deployment

if __name__ == "__main__":
    try:
        logger.info(start_stage_logger(STAGE_NAME))
        EvaluateModelPipeline.main()
        logger.info(end_stage_logger(STAGE_NAME))
    except Exception as e:
        logger.exception(e)
        raise e