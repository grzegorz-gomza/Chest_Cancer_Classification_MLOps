from ChestCancerClassifier import logger
from ChestCancerClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from ChestCancerClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelPipeline
from ChestCancerClassifier.pipeline.stage_03_train_model import TrainModelPipeline
from ChestCancerClassifier.pipeline.stage_04_evaluate_model import EvaluateModelPipeline

from ChestCancerClassifier.utils.common import start_stage_logger, end_stage_logger

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


STAGE_NAME = "Model Fine-Tuning"
try:
    logger.info(start_stage_logger(STAGE_NAME))
    TrainModelPipeline.main()
    logger.info(end_stage_logger(STAGE_NAME))
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Evaluate Model"
try:
    logger.info(start_stage_logger(STAGE_NAME))
    EvaluateModelPipeline.main()
    logger.info(end_stage_logger(STAGE_NAME))
except Exception as e:
    logger.exception(e)
    raise e