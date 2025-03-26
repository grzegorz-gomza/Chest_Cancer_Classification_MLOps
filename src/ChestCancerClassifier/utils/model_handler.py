import tensorflow as tf
import numpy as np
from PIL import Image

from ChestCancerClassifier.entity.config_entity import EvaluationConfig

class ModelHandler:
    def __init__(self,
                 config: EvaluationConfig):
        # Load your TensorFlow model here
        self.config = config
        self.model = tf.keras.models.load_model(self.config.model_path)

    def preprocess_image(self, image_path):
        processed_image = tf.io.read_file(image_path)
        processed_image = tf.image.decode_image(processed_image, channels=self.config.params_image_size[2])
        if processed_image.shape.as_list() != self.config.params_image_size:
            processed_image = tf.image.resize(processed_image, self.config.params_image_size[:2])
        processed_image = tf.expand_dims(processed_image, axis=0)
        processed_image = tf.cast(processed_image, tf.float32) / 255.0
        
        
        return processed_image

    def predict(self, image_path):
        processed_image = self.preprocess_image(image_path)
        prediction = self.model.predict(processed_image)
        return self.format_prediction(prediction)

    def predict_sample(self, sample_name):
        image_path = f"static/images/samples/{sample_name}"
        return self.predict(image_path)

    def format_prediction(self, prediction):
        # Format your prediction results here
        return str(prediction)