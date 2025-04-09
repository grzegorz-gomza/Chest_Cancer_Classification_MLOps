import tensorflow as tf
from ChestCancerClassifier.entity.config_entity import PrepareBaseModelConfig

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

        self.model = tf.keras.applications.VGG16(
            input_shape = self.config.params_image_size,
            weights = self.config.params_weights,
            include_top = self.config.params_include_top
            )

        # save the model
        self.model.save(self.config.base_model_path)

    @staticmethod
    def _prepare_full_model(model, config, classes, freeze_all, freeze_till, learning_rate):
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
        # Create the data augmentation layers
        inputs = tf.keras.layers.Input(shape = config.params_image_size, name = "new_input_layer")
        
        # Data augmentation layers
        augmentation_layers = tf.keras.Sequential([
            tf.keras.layers.RandomFlip(config.params_random_flip),
            tf.keras.layers.RandomRotation(config.params_random_rotation),
            tf.keras.layers.RandomZoom(config.params_random_zoom),
            tf.keras.layers.RandomContrast(config.params_random_contrast),
            tf.keras.layers.RandomBrightness(config.params_random_brightness),
        ], name = "image_augmentation")
        
        # Apply augmentation only during training
        augmented = augmentation_layers(inputs, training=True)

        # Freeze the model
        if freeze_all:
            for layer in model.layers:
                model.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                model.trainable = False

        # Connect the augmentation output to the VGG16 base
        if config.params_augmentation:
            x = model(augmented, training=False)

        # Add custom layers for fine-tuning
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        x = tf.keras.layers.Dense(1024, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        x = tf.keras.layers.Dense(1024, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        x = tf.keras.layers.Dense(1024, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        x = tf.keras.layers.Dense(1024, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        x = tf.keras.layers.Dense(1024, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        x = tf.keras.layers.Dense(258, activation='relu')(x)
        outputs = tf.keras.layers.Dense(
                    units = classes,
                    activation = "softmax",
                    name = "output_layer"
                    )(x)
        
        full_model = tf.keras.models.Model(
            inputs = inputs,
            outputs = outputs
            )

        if config.params_use_categorical_encoding:
            loss_function = tf.keras.losses.CategoricalCrossentropy()
        else:
            loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
            
        full_model.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss = loss_function,
            metrics = ['accuracy', 'auc']
        )

        full_model.summary()
        return full_model

    
    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            config=self.config,
            classes=self.config.params_classes,
            freeze_all=False,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )
        
        self.full_model.save(self.config.updated_base_model_path)
