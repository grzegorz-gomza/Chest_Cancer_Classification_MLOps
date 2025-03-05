import os

import tensorflow as tf

from ChestCancerClassifier.entity.config_entity import TrainModelConfig
from ChestCancerClassifier.components.image_preprocess import ImagePreprocessDataSplitter
from ChestCancerClassifier import logger

class TrainModel:
    def __init__(self,
                config: TrainModelConfig,
                ):
        self.config = config
        self.data = ImagePreprocessDataSplitter(data_dir = self.config.ingested_data_path,
                        config = config
                        )

    def _get_base_model(self):
        if os.path.exists(self.config.model_path) and self.config.params_use_pretrained_model:
            self.model = tf.keras.models.load_model(
                self.config.model_path)
        else:
            self.model = tf.keras.models.load_model(
                    self.config.updated_base_model_path
                )
        
    def _data_split(self, dataset):
        X_train, X_test, X_val, y_train, y_test, y_val, class_labels = self.data.create_data_with_split()
        batch_size = self.config.params_batch_size
        shapes = (X_train.shape[0], X_test.shape[0], X_val.shape[0], y_train.shape[0], y_test.shape[0], y_val.shape[0],)
        if dataset == True:
            train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
            train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE).cache().repeat()

            val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
            val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE).cache()

            test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
            test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE).cache()

            return train_dataset, test_dataset, val_dataset, class_labels, shapes
        else:
            return X_train, X_test, X_val, y_train, y_test, y_val, class_labels, shapes
        
    def train_model(self):
        dataset = False
        
        # get the base model
        self._get_base_model()

        # get the train test val data sets
        if dataset:
            train_dataset, test_dataset, val_dataset, class_labels, shapes = self._data_split(dataset)
            self.steps_per_epoch = shapes[0] // self.config.params_batch_size
            self.validation_steps = shapes[2]// self.config.params_batch_size
        else:
            X_train, X_test, X_val, y_train, y_test, y_val, class_labels, shapes = self._data_split(dataset)
            self.steps_per_epoch = None
            self.validation_steps = None


        # Train the model
        self.model.fit(x = X_train if not dataset else train_dataset,
                       y = y_train if not dataset else None,
                       batch_size = self.config.params_batch_size,
                       epochs = self.config.params_epochs,
                       verbose = 2,
                       validation_data = (X_val, y_val) if not dataset else val_dataset,                       
                       shuffle = True,
                       steps_per_epoch = self.steps_per_epoch,
                       validation_steps = self.validation_steps)

        # Save the model
        self.model.save(self.config.model_path)