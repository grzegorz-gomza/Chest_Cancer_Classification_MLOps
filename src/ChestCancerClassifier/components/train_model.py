import os
import random
import numpy as np
from PIL import Image
from pathlib import Path

from sklearn.model_selection import train_test_split
import tensorflow as tf

from ChestCancerClassifier.entity.config_entity import TrainModelConfig
from ChestCancerClassifier import logger

class ImageProcessor:
    def __init__(self,
                config: TrainModelConfig
                ):
        self.config = config

    def list_paths(self, img_dir_path: Path) -> list[Path]:
        """
        Generates a list of paths to all files in a given folder.

        This function traverses the specified folder and all its subfolders,
        then creates a list of full paths to the files found therein.

        Args:
            folder (Path): The path to the folder that needs to be searched.

        Returns:
            list: A list of full paths to the files located in the given folder and its subfolders.
        """
        paths = []
        
        for root, dirs, files in os.walk(img_dir_path):
            for file in files:
                paths.append(os.path.join(root, file))
    
        return paths

    def take_sample(self, path: list[Path], percent: int) -> list [Path]:
        """
        Takes a random sample of paths based on a given percentage.

        Args:
            path (list): The list of file paths.
            percent (int): The percentage of files to be selected.

        Returns:
            list: A list containing the randomly selected paths.
        """
        if percent < 0 or percent > 100:
            raise ValueError("The percentage must be an integer number in the range of 0 to 100")
        
        elem_nums = int(len(path) * (percent / 100))
        chosen_paths = random.sample(path, elem_nums)
        
        return chosen_paths

    def retrieve_num_subdirectories(folder: Path, num_classes: int, folder_deep: int = 0) -> dict[str: Path]:
        """
        Returns a dict of the first num_classes directories at a given depth.

        Args:
            folder (str): The root folder to start the search from.
            num_classes (int): The number of directories to return.
            folder_deep (int): The depth of subfolders to be considered.

        Returns:
            dict: A dictionary containing class name as a key and directory path as value
        """
        dir_dict = {}
        for i, (root, dirs, files) in enumerate(os.walk(folder)):
            if i == folder_deep:
                for j, dir in enumerate(dirs):
                    dir_dict[dir] = os.path.join(root,dir)
                    # dir_list.append(os.path.join(root, dir))
                    if j >= (num_classes - 1):
                        break
            else:
                continue
        
        return dir_dict

    def process_data(self,
                    folder: Path,
                    percent: int,
                    take_sample_of: str,
                    num_classes: int = None,
                    folder_deep: int = 0) -> list[Path]:
        """
        Processes data by either taking a sample from the dataset or restricting to a certain number of classes.

        Args:
            folder (Path): The path to the main folder containing the data.
            percent (int): The percentage of files to be selected.
            take_sample_of (str): The criteria to take sample from ('dataset' or 'classes').
            num_classes (int, optional): The number of classes to be considered if take_sample_of is 'classes'. Default is None.
            folder_deep (int, optional): The depth to which classes are considered. Default is 0.

        Returns:
            list: A list of paths to the selected files.
        """
        if take_sample_of == 'dataset':
            paths = self.list_paths(folder)
            sampled_paths = self.take_sample(paths, percent)
            return sampled_paths
        
        elif take_sample_of == 'classes':
            if num_classes is None:
                raise ValueError("num_classes must be provided when take_sample_of is 'classes'")
            
            class_folders = self.retrieve_num_subdirectories(folder, num_classes, folder_deep)
            class_paths = []
            
            for _, class_folder in class_folders.items():
                paths = self.list_paths(class_folder)
                class_paths.extend(paths)
            if percent < 100:
                sampled_paths = self.take_sample(class_paths, percent)
                return sampled_paths
            else: return class_paths
        
        else:
            raise ValueError("take_sample_of must be either 'dataset' or 'classes'")

    def create_data_with_split(self,
                                image_paths: list[Path],
                                image_size: list[int],
                                val_size: float = 0.2,
                                test_size: float = 0.2) -> tuple:
        """
        Creates a dataset of images and their corresponding labels from a list of image paths.

        To use in case of not having a predefiend train, test and validation sets.
        If it's the case --> use create_data_without_split

        It assumses that the images are in folders named as the class names.
        
        This function preprocesses the images, extracting their labels from the subdirectory names,
        and splits the data into training, validation, and testing sets based on the specified sizes.

        Args:
            image_paths (list of str): A list of file paths to the images.
            image_size (list of int): A list of image dimentions
            val_size (float): Proportion of the training data to be used as validation data (default is 0.2).
            test_size (float): Proportion of the data to be used as the test set (default is 0.2).

            Returns:
                tuple: (X_train, X_val, X_test, y_train, y_val, y_test) as tf.float32 tensors.

        Notes:
            - The function expects that the labels are defined by the subdirectory names of the images.
            - If test_size is set to 0, the function will return all images and their labels without splitting.
        """
        # Console log
        logger.info(f'Starting Image preprocess stream...')
        logger.info(f'Number of images to process: {len(image_paths)}')
        
        # Get parameters of image size
        image_size = self.config.params_image_size
        
        # Initialize tensors
        X = tf.zeros((0, img_height, img_weight, num_channels), dtype=tf.float32)
        y = tf.zeros((0,), dtype=tf.float32)

        # Image size validation
        try:
            sample_img = tf.io.read_file(image_paths[0])
        except Exception as e:
            logger.info(f'Error trying to recieve an image from a path: {image_paths[0]}')
            raise e

        try:
            num_channels = image_size[2]
            sample_img = tf.image.decode_image(image, channels=num_channels)
        except Exception as e:
            logger.info(f'Error tryining to decode an image from a path: {image_paths[0]} with {num_channels}\nThe number of channels in the given Image does not correspond with definition in params.yaml')
            raise e

        # Check if resize is needed
        sample_img = tf.cast(image, tf.float32) / 255.0
        sample_img_shape = sample_img.numpy().shape
        img_height = image_size[0]
        img_weight = image_size[1]
        
        if sample_img_shape == (img_height, img_weight, num_channels):
            resize = False
        else:
            resize = True
            logger.info(f'The image size does not match the size given in project parameters.\nThe images are going to be resized')

        #######################################
        # Main loop
        for path in image_paths:
            # Read and preprocess the image
            image = tf.io.read_file(path)
            image = tf.image.decode_image(image, channels=num_channels)
            if resize:
                image = tf.image.resize(image, [img_height, img_weight, num_channels])
            image = tf.cast(image, tf.float32) / 255.0
            
            # Extract the label from the subdirectory name
            label = os.path.basename(os.path.dirname(path))
            
            # Append to tensors
            X = tf.concat([X, tf.expand_dims(image, axis=0)], axis=0)
            y = tf.concat([y, tf.expand_dims(tf.constant([label], dtype=tf.float32), axis=0)], axis=0)

        # Split data
        if test_size != 0 and val_size != 0:
            # Split into training and testing sets
            total = len(image_paths)
            test_idx = int(total * test_size)
            val_idx = int(total * val_size)
            train_idx = total - test_idx - val_idx

            X_train, X_test, X_val = X[:train_idx], X[train_idx:train_idx+test_idx], X[train_idx+test_idx:]
            y_train, y_test, y_val = y[:train_idx], y[train_idx:train_idx+test_idx], y[train_idx+test_idx:]

            return X_train, X_test, X_val, y_train, y_test, y_val
            
        elif test_size != 0 and val_size == 0:
            # Split into training and testing sets
            total = len(image_paths)
            test_idx = int(total * test_size)
            train_idx = total - test_idx

            X_train, X_test = X[:train_idx], X[train_idx:train_idx+test_idx]
            y_train, y_test = y[:train_idx], y[train_idx:train_idx+test_idx]

            return X_train, X_test, y_train, y_test
        else:
            return X, y

    def create_data_without_split(self,
                                    image_paths: list[Path],
                                    image_size: list[int]) -> tuple:
            """
            Creates a dataset of images and their corresponding labels from a list of image paths.
 
            To use in case of having a predefiend train, test and validation sets.
            If it's the case --> use create_data_with_split

            It assumses that the images are in folders: train, test, valid
            and the pictures are distributed over folders containing class label.
            
            This function preprocesses the images, extracting their labels from the subdirectory names,
            and splits the data into training, validation, and testing sets based on the specified sizes.

            Args:
                image_paths (list of Path): A list of file paths to the images.
                image_size (list of int): A list of image dimentions
                val_size (float): Proportion of the training data to be used as validation data (default is 0.2).
                test_size (float): Proportion of the data to be used as the test set (default is 0.2).

            Returns:
                tuple: (X_train, X_val, X_test, y_train, y_val, y_test) as tf.float32 tensors.

            Notes:
                - The function expects that the labels are defined by the subdirectory names of the images.
                - If test_size is set to 0, the function will return all images and their labels without splitting.
            """
            # Console log
            logger.info(f'Starting Image preprocess stream...')
            logger.info(f'Number of images to process: {len(image_paths)}')
            
            # Get parameters of image size
            image_size = self.config.params_image_size

            # Image size validation
            try:
                sample_img = tf.io.read_file(image_paths[0])
            except Exception as e:
                logger.info(f'Error trying to recieve an image from a path: {image_paths[0]}')
                raise e

            try:
                num_channels = image_size[2]
                sample_img = tf.image.decode_image(image, channels=num_channels)
            except Exception as e:
                logger.info(f'Error tryining to decode an image from a path: {image_paths[0]} with {num_channels}\nThe number of channels in the given Image does not correspond with definition in params.yaml')
                raise e

            # Check if resize is needed
            sample_img = tf.cast(image, tf.float32) / 255.0
            sample_img_shape = sample_img.numpy().shape
            img_height = image_size[0]
            img_weight = image_size[1]
            
            if sample_img_shape == (img_height, img_weight, num_channels):
                resize = False
            else:
                resize = True
                logger.info(f'The image size does not match the size given in project parameters.\nThe images are going to be resized')

            #######################################
            # Initialize tensors
            X = tf.zeros((0, img_height, img_weight, num_channels), dtype=tf.float32)
            y = tf.zeros((0,), dtype=tf.float32)
            
            # Main loop
            for path in image_paths:
                # Read and preprocess the image
                image = tf.io.read_file(path)
                image = tf.image.decode_image(image, channels=num_channels)
                if resize:
                    image = tf.image.resize(image, [img_height, img_weight, num_channels])
                image = tf.cast(image, tf.float32) / 255.0
                
                # Extract the label from the subdirectory name
                label = os.path.basename(os.path.dirname(path))
                
                # Append to tensors
                X = tf.concat([X, tf.expand_dims(image, axis=0)], axis=0)
                y = tf.concat([y, tf.expand_dims(tf.constant([label], dtype=tf.float32), axis=0)], axis=0)

            # Split data
            if test_size != 0 and val_size != 0:
                # Split into training and testing sets
                total = len(image_paths)
                test_idx = int(total * test_size)
                val_idx = int(total * val_size)
                train_idx = total - test_idx - val_idx

                X_train, X_test, X_val = X[:train_idx], X[train_idx:train_idx+test_idx], X[train_idx+test_idx:]
                y_train, y_test, y_val = y[:train_idx], y[train_idx:train_idx+test_idx], y[train_idx+test_idx:]

                return X_train, X_test, X_val, y_train, y_test, y_val
                
            elif test_size != 0 and val_size == 0:
                # Split into training and testing sets
                total = len(image_paths)
                test_idx = int(total * test_size)
                train_idx = total - test_idx

                X_train, X_test = X[:train_idx], X[train_idx:train_idx+test_idx]
                y_train, y_test = y[:train_idx], y[train_idx:train_idx+test_idx]

                return X_train, X_test, y_train, y_test
            else:
                return X, y


class TrainModel:
    def __init__(self,
                config: TrainModelConfig
                ):
        self.config = config

    def create_train_test_data(self):

        
    def train_model(self, model,):
        model.fit(
            )