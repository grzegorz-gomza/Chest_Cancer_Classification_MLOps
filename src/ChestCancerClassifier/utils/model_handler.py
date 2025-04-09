import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path
import tensorflow as tf


from ChestCancerClassifier.entity.config_entity import EvaluationConfig

class ModelHandler:
    def __init__(self, config: EvaluationConfig):
        # Load your TensorFlow model here
        self.config = config
        self.model = tf.keras.models.load_model(self.config.model_path_deploy)

    def preprocess_image(self, image):
        """
        Preprocesses the image data.  Expects a NumPy array.
        """
        processed_image = Image.fromarray(image)
        processed_image = processed_image.resize(self.config.params_image_size[:2])
        processed_image = np.array(processed_image)
        if processed_image.shape[-1] == 4: #Check for alpha channel
            processed_image = processed_image[:,:,:3] #Remove alpha channel if present
        processed_image = np.expand_dims(processed_image, axis=0)
        processed_image = processed_image.astype(np.float32) / 255.0
        return processed_image
    def predict(self, image):

        processed_image = self.preprocess_image(image)
        prediction = self.model.predict(processed_image)
        return prediction

    def predict_endpoint(self, image):
        processed_image = self.preprocess_image(image)
        payload = processed_image.tobytes()
        response = self.runtime_client.invoke_endpoint(EndpointName=self.end_point,
                                                       ContentType='image/jpeg',
                                                       Body=payload)
        prediction = np.frombuffer(response['Body'].read(), dtype=np.float32)

