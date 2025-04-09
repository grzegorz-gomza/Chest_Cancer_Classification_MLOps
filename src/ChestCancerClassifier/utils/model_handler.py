import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path
import tensorflow as tf
import os


from ChestCancerClassifier.entity.config_entity import EvaluationConfig

class ModelHandler:
    def __init__(self, config: EvaluationConfig):
        # Load your TensorFlow model here
        self.config = config
        try:
            self.model = tf.keras.models.load_model(self.config.model_path_deploy)
        except Exception as e:
            print(f"Erorr by reading the model file\n")
            print(f"Current working directory: {os.getcwd()}\n")
            print(f"Attempting to load model from: {self.config.model_path_deploy}\n")
            print(f"Does file exist? {os.path.exists(self.config.model_path_deploy)}\n")
            os.chdir('/app')
            print(f"Directory changed to /app, current working dir: {os.getcwd()}")


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

