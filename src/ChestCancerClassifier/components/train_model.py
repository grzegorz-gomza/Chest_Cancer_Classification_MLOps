import os
import random
import numpy as np
from pathlib import Path

import tensorflow as tf

from ChestCancerClassifier.entity.config_entity import TrainModelConfig
from ChestCancerClassifier import logger

class TrainModel:
    def __init__(self,
                config: TrainModelConfig
                ):
        self.config = config

    def get_base_model(self):
        self.model = tf.keras.models.load_model(
                self.config.updated_base_model_path
            )
        
        
    def train_model(self, model):
        model.fit(
            )