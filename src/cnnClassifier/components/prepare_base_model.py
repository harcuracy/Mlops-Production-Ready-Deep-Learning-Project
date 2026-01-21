import os
import tensorflow as tf
from pathlib import Path

from cnnClassifier.entity.config_entity import PrepareBaseModelConfig


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
        self.model = None
        self.full_model = None

    def get_base_model(self):
        """
        Load the ResNet50 base model with ImageNet weights
        """
        self.model = tf.keras.applications.ResNet50(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )

        self.save_model(
            path=self.config.base_model_path,
            model=self.model
        )

    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        """
        Prepare the full model by:
        - Freezing layers
        - Adding custom classification head
        - Compiling the model
        """

        # Freeze base model layers
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False

        elif freeze_till is not None and freeze_till > 0:
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False

        # Custom classification head
        x = tf.keras.layers.GlobalAveragePooling2D()(model.output)
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        prediction = tf.keras.layers.Dense(classes, activation="softmax")(x)

        # Final model
        full_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=prediction
        )

        # Compile model
        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        full_model.summary()
        return full_model

    def update_base_model(self):
        """
        Create and save the full transfer-learning model
        """
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )

        self.save_model(
            path=self.config.updated_base_model_path,
            model=self.full_model
        )

    @staticmethod
    def save_model(path, model: tf.keras.Model):
        """
        Save model to disk
        """
        model.save(path)
