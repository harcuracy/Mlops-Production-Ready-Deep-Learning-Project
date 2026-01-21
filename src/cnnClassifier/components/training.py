import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
import time
from pathlib import Path


from cnnClassifier.entity.config_entity import TrainingConfig
from cnnClassifier import logger









class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.class_indices = None
        self.index_to_class = None

    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )

    def train_valid_generator(self):

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        datagenerator_kwargs = dict(
            preprocessing_function=preprocess_input,
            validation_split=0.20
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=10,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

    
        self.class_indices = self.train_generator.class_indices
        self.index_to_class = {v: k for k, v in self.class_indices.items()}

        self.show_class_labels()

    def train(self):

        self.steps_per_epoch = (
            self.train_generator.samples // self.train_generator.batch_size
        )

        self.validation_steps = (
            self.valid_generator.samples // self.valid_generator.batch_size
        )

        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_data=self.valid_generator,
            validation_steps=self.validation_steps
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )

    def show_class_labels(self):
        logger.info("Class indices (folder name to index):")
        logger.info(self.class_indices)

        logger.info("Index to class mapping:")
        logger.info(self.index_to_class)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
