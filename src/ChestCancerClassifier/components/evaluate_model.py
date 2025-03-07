import os
import numpy as np
from dotenv import load_dotenv
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
import mlflow
import dagshub
from urllib.parse import urlparse

from ChestCancerClassifier import logger
from ChestCancerClassifier.entity.config_entity import EvaluationConfig
from ChestCancerClassifier.components.train_model import TrainModel
from ChestCancerClassifier.utils.common import load_from_pickle, save_json

load_dotenv()
os.getenv("MLFLOW_TRACKING_USERNAME")
os.getenv("MLFLOW_TRACKING_PASSWORD")
# mlflow.autolog()

class EvaluateModel:
    def __init__(self,
                 config: EvaluationConfig):
        self.config = config

    def _get_test_dataset(self):
        # Set the appropriate data directory based on the dataset type
        data_type_dir = "categorical" if self.config.params_use_categorical_encoding else "numerical"
        load_path = self.config.pickle_data_path/ "preprocessed_data" / data_type_dir
        
        # Load the pickle files with appropriate file names
        file_suffix = "cat" if self.config.params_use_categorical_encoding else "num"
        X_test, y_test = load_from_pickle(load_path, f"test_set_{file_suffix}.pkl")
        class_labels = load_from_pickle(load_path, f"class_labels_{file_suffix}.pkl")
        return X_test, y_test, class_labels

    def evaluate(self):
        self.model = load_model(self.config.model_path)
        self.X_test, self.y_test, self.class_labels = self._get_test_dataset()
        
        if self.config.params_use_tf_dataset:
            batch_size = self.config.params_batch_size
            # Create and configure datasets
            self.test_dataset = TrainModel.create_dataset(self.X_test, self.y_test, batch_size)

        self.score = self.model.evaluate(x = self.X_test if not self.config.params_use_tf_dataset else self.test_dataset,
                                    y = self.y_test if not self.config.params_use_tf_dataset else None,
                                    verbose = 2)

    def save_score(self):
        self.scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path = self.config.root_dir / "scores.json", data = self.scores)
        logger.info(f"Model evaluated: loss: {self.score[0]}, accuracy: {self.score[1]}")

    def mlflow_tracking(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme # retrieves the first part of url (https://)
        # dagshub.init(repo_owner='grzegorz-gomza', repo_name='Chest_Cancer_Classification_MLOps', mlflow=True)
        run = mlflow.active_run()
        if  any(run.info.status == value for value in [True, "RUNNING"]):
            logger.info(f"run_id: {run.info.run_id}; status: {run.info.status}")
            mlflow.end_run()
            logger.info(f"run_id: {run.info.run_id}; status: {run.info.status}")

        
        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics({"loss": self.scores["loss"],
                            "accuracy": self.scores["accuracy"]})
            
            if tracking_url_type_store != "file":
                mlflow.keras.log_model(
                    self.model, 
                    "model", 
                    registered_model_name="VGG16Model",
                )
            else:
                mlflow.keras.log_model(
                    self.model, 
                    "model"
                )
            mlflow.end_run()

