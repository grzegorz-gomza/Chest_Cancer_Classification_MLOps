import os
from pathlib import Path
import tensorflow as tf
from ChestCancerClassifier.entity.config_entity import TrainModelConfig
from ChestCancerClassifier import logger
from tqdm import tqdm

class ImagePreprocessDataSplitter:
    def __init__(self,
                 data_dir: Path,
                 config: TrainModelConfig):
        """
        Initialization with data directory path and configuration provided.

        :param data_dir: The Path object of the root directory containing the dataset.
        :param config: Configuration object for model training.
        """
        self.data_dir = data_dir
        self.config = config
        
    def create_data_with_split(self, categorical: bool = True):
        """
        Creates and preprocesses datasets for training, testing, and validation.

        This function iterates over a directory structure containing 'train', 'test', and 'valid' folders, each with subfolders 
        representing classes. It reads images from these class folders, decodes, resizes, and normalizes them, and assigns 
        integer labels based on subfolder names. The images and labels are then split into training, testing, and validation 
        datasets.

        Takes:
        :param categorical: If True, the one-hot encoding of the labels is returned.
                            If False, the integer labels are returned.
        Returns:
            tuple: A tuple containing the following elements:
                - X_train: A tf.float32 tensor of training images.
                - X_test: A tf.float32 tensor of testing images.
                - X_val: A tf.float32 tensor of validation images.
                - y_train: A tf.int32 tensor of labels corresponding to the training images.
                - y_test: A tf.int32 tensor of labels corresponding to the testing images.
                - y_val: A tf.int32 tensor of labels corresponding to the validation images.
                - class_labels: A dictionary mapping class names to integer labels.
        """
        logger.info(f"Start image pre-processing")
        img_size = self.config.params_image_size

        # Initialize tensors
        X_train = tf.zeros((0, img_size[0], img_size[1], img_size[2]), dtype=tf.float32)
        X_test = tf.zeros((0, img_size[0], img_size[1], img_size[2]), dtype=tf.float32)
        X_val = tf.zeros((0, img_size[0], img_size[1], img_size[2]), dtype=tf.float32)

        y_train = tf.zeros((0,), dtype=tf.int32)
        y_test = tf.zeros((0,), dtype=tf.int32)
        y_val = tf.zeros((0,), dtype=tf.int32)

        # Prepare temp data structures
        class_labels = {}
        label_counter = 0
        resize = None

        # Main loop over the source dir
        for root, dirs, files in os.walk(self.data_dir):
            # Adjust the list if folder names are different. 
            # Assuring, that loop is in the correct folder
            if any(folder in dirs for folder in ["train", "test", "valid"]):
                # Loop around train test valid
                for folder in dirs:
                    folder_path = os.path.join(root, folder)
                    # Loop around classes
                    for class_folder in os.listdir(folder_path):
                        class_path = os.path.join(folder_path, class_folder)
                        # Labeling the classes
                        if class_folder not in class_labels:
                            class_labels[class_folder] = label_counter
                            label_counter += 1
                        # Image processing
                        for file in tqdm(os.listdir(class_path)):
                            file_path = os.path.join(class_path, file)
                            image = tf.io.read_file(file_path)
                            image = tf.image.decode_image(image, channels=img_size[2])
                            # Assuring that the size matches the CNN input
                            if resize == None:
                                if image.shape.as_list() == img_size:
                                    resize = False
                                else:
                                    resize = True
                                    logger.info(f"Image size has to be adjusted from {image.shape.as_list()} into {img_size}")
                            image = tf.image.resize(image, img_size[:2])
                            image = tf.cast(image, tf.float32) / 255.0
                            # Spliting the images
                            if folder == "train":
                                X_train = tf.concat([X_train, tf.expand_dims(image, axis=0)], axis=0)
                                y_train = tf.concat([y_train, tf.constant([class_labels[class_folder]], dtype=tf.int32)], axis=0)
                            elif folder == "test":
                                X_test = tf.concat([X_test, tf.expand_dims(image, axis=0)], axis=0)
                                y_test = tf.concat([y_test, tf.constant([class_labels[class_folder]], dtype=tf.int32)], axis=0)
                            elif folder == "valid":
                                X_val = tf.concat([X_val, tf.expand_dims(image, axis=0)], axis=0)
                                y_val = tf.concat([y_val, tf.constant([class_labels[class_folder]], dtype=tf.int32)], axis=0)

        if categorical:
            y_train = tf.keras.utils.to_categorical(y_train, num_classes=self.config.params_classes)
            y_test = tf.keras.utils.to_categorical(y_test, num_classes=self.config.params_classes)
            y_val = tf.keras.utils.to_categorical(y_val, num_classes=self.config.params_classes)
            
        logger.info(f"Image preprocess done")
        return X_train, X_test, X_val, y_train, y_test, y_val, class_labels
