import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf

from BreastCancerClassifier.entity.config_entity import PrepareBaseModelConfig

class PrepareBaseModel:
    def __init__(self,
                 config: PrepareBaseModelConfig
                 ):
        """
        Initialization with configuration provided.

        :param config: Configuration object for preparing base model.
        """
        self.config = config

    def download_base_model(self):
        # Get and download keras VGG16 model
        """
        Downloads the Keras VGG16 model with specified configurations and saves it.

        The method utilizes the Keras VGG16 pre-trained model. It configures the 
        model with parameters such as input shape, weights, and whether to include 
        the fully connected top layers. After downloading, the model is saved to 
        the specified path in the configuration.
        """

        self.model = tf.keras.applications.vgg16.VGG16(
            input_shape = self.config.params_image_size,
            weights = self.config.params_weights,
            include_top = self.config.params_include_top
            )

        # save the model
        self.model.save(self.config.base_model_path)

    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        """
        Prepares the full model for training by freezing specified layers and 
        adding a dense output layer.

        Args:
            model (tf.keras.Model): The base model to be modified.
            classes (int): Number of output classes for the dense layer.
            freeze_all (bool): If True, all layers in the model are frozen.
            freeze_till (int or None): Number of layers from the end to remain trainable. 
                                    If None or 0, no layers are frozen.
            learning_rate (float): Learning rate for the SGD optimizer.

        Returns:
            tf.keras.Model: The compiled full model ready for training.
        """

        if freeze_all:
            for layer in model.layers:
                model.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                model.trainable = False

        flatten_in = tf.keras.layers.Flatten()(model.output)
        prediction = tf.keras.layers.Dense(
            units = classes,
            activation = "softmax"
            )(flatten_in)
        
        full_model = tf.keras.models.Model(
            inputs = model.input,
            outputs = prediction
            )

        full_model.compile(
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate),
            loss = tf.keras.losses.CategoricalCrossentropy(),
            metrics = ['accuracy']
            )

        full_model.summary()
        return full_model

    
    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )
        
        self.full_model.save(self.config.updated_base_model_path)
