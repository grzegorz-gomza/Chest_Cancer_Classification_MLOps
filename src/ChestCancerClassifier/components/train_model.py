import os

import tensorflow as tf

from ChestCancerClassifier.entity.config_entity import TrainModelConfig
from ChestCancerClassifier.components.image_preprocess import ImagePreprocessDataSplitter
from ChestCancerClassifier import logger
from ChestCancerClassifier.utils.common import load_from_pickle

class TrainModel:
    def __init__(self,
                config: TrainModelConfig,
                ):
        self.config = config
        self.use_dataset = config.params_use_tf_dataset
        self.use_categorical_encoding = config.params_use_categorical_encoding
        self.use_pickled_data = config.params_use_pickled_data
        self.use_pretrained_model = config.params_use_pretrained_model
        self.data = ImagePreprocessDataSplitter(data_dir = self.config.ingested_data_path,
                        config = config
                        )

    def _get_base_model(self):
        if os.path.exists(self.config.model_path) and self.use_pretrained_model:
            self.model = tf.keras.models.load_model(
                self.config.model_path)
            self.model.summary()
        else:
            self.model = tf.keras.models.load_model(
                    self.config.updated_base_model_path
                )

    def _data_split(self):
        """
        Load or create data splits and optionally convert to TensorFlow datasets.
        
        Args:
            use_dataset (bool): Whether to return TensorFlow datasets (True) or raw arrays (False).
            
        Returns:
            tuple: Either (train_dataset, test_dataset, val_dataset, class_labels, shapes) if use_dataset=True
                or (X_train, X_test, X_val, y_train, y_test, y_val, class_labels, shapes) if use_dataset=False
        """
        # Determine if we should load pickled data or create new data
        if os.path.exists(self.config.root_dir / "preprocessed_data") and self.use_pickled_data:
            # Set the appropriate data directory based on the dataset type
            data_type_dir = "categorical" if self.use_categorical_encoding else "numerical"
            load_path = self.config.root_dir / "preprocessed_data" / data_type_dir
            
            # Load the pickle files with appropriate file names
            file_suffix = "cat" if self.use_categorical_encoding else "num"
            X_train, y_train = load_from_pickle(load_path, f"train_set_{file_suffix}.pkl")
            X_test, y_test = load_from_pickle(load_path, f"test_set_{file_suffix}.pkl")
            X_val, y_val = load_from_pickle(load_path, f"valid_set_{file_suffix}.pkl")
            class_labels = load_from_pickle(load_path, f"class_labels_{file_suffix}.pkl")
        else:
            # Create new data splits
            X_train, X_test, X_val, y_train, y_test, y_val, class_labels = self.data.create_data_with_split()
        
        # Calculate shapes regardless of data source
        shapes = (X_train.shape[0], X_test.shape[0], X_val.shape[0], 
                y_train.shape[0], y_test.shape[0], y_val.shape[0])
        
        # Return either TensorFlow datasets or raw arrays based on use_dataset flag
        if self.use_dataset:
            batch_size = self.config.params_batch_size
            
            # Create and configure datasets
            train_dataset = self.create_dataset(X_train, y_train, batch_size, repeat=True)
            val_dataset = self.create_dataset(X_val, y_val, batch_size)
            test_dataset = self.create_dataset(X_test, y_test, batch_size)
            
            return train_dataset, test_dataset, val_dataset, class_labels, shapes
        else:
            return X_train, X_test, X_val, y_train, y_test, y_val, class_labels, shapes

    @staticmethod
    def create_dataset(features, labels, batch_size, repeat=False):
        """
        Create a TensorFlow dataset from features and labels.
        
        Args:
            features: Input features
            labels: Target labels
            batch_size (int): Batch size for the dataset
            repeat (bool): Whether to repeat the dataset indefinitely
            
        Returns:
            tf.data.Dataset: Configured TensorFlow dataset
        """
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE).cache()
        
        if repeat:
            dataset = dataset.repeat()
            
        return dataset

    def train_model(self):
        # get the base model
        self._get_base_model()

        # get the train test val data sets
        if self.use_dataset:
            train_dataset, test_dataset, val_dataset, class_labels, shapes = self._data_split()
            self.steps_per_epoch = shapes[0] // self.config.params_batch_size
            self.validation_steps = shapes[2]// self.config.params_batch_size
        else:
            X_train, X_test, X_val, y_train, y_test, y_val, class_labels, shapes = self._data_split()
            self.steps_per_epoch = None
            self.validation_steps = None

            EARLY_STOPPING = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", restore_best_weights=True,
            # start_from_epoch = 5,
            patience = 100
        )

        def scheduler(epoch, lr):          
            if epoch < 50:
                return 0.001
            elif epoch < 150:
                return 0.0001
            elif epoch < 250:
                return 0.00001
            elif epoch < 350:
                return 0.000001
            elif epoch < 450:
                return 0.0000001
            elif epoch < 550:
                return 0.00000001
            elif epoch < 650:
                return 0.000000001
            elif epoch < 750:
                return 0.0000000001 
            else:
                return 0.00000000001

     

        # Create a learning rate scheduler
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)

        # Compile the model
        # Train the model
        self.model.fit(x = X_train if not self.use_dataset else train_dataset,
                       y = y_train if not self.use_dataset else None,
                       batch_size = self.config.params_batch_size,
                       epochs = self.config.params_epochs,
                       verbose = 2,
                       validation_data = (X_val, y_val) if not self.use_dataset else val_dataset,
                       shuffle = True,
                       steps_per_epoch = self.steps_per_epoch,
                       validation_steps = self.validation_steps,
                    #    callbacks = [
                    #     #    EARLY_STOPPING,
                    #     #    lr_scheduler
                    #     ]
                       )

        # Save the model
        self.model.save(self.config.model_path)