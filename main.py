from BreastCancerClassifier import logger
from BreastCancerClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from BreastCancerClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelPipeline
from BreastCancerClassifier.utils.common import start_stage_logger, end_stage_logger

STAGE_NAME = "Data Ingestion stage"

try:
    logger.info(start_stage_logger(STAGE_NAME))
    DataIngestionTrainingPipeline.main()
    logger.info(end_stage_logger(STAGE_NAME))
except Exception as e:
    logger.exception(e)
    raise e



STAGE_NAME = "Prepare Base Model stage"
try:
    logger.info(start_stage_logger(STAGE_NAME))
    PrepareBaseModelPipeline.main()
    logger.info(end_stage_logger(STAGE_NAME))
except Exception as e:
    logger.exception(e)
    raise e