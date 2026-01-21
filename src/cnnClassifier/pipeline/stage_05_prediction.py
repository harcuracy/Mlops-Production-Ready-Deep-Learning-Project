import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

from cnnClassifier import logger





class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        # Load trained model
        model = tf.keras.models.load_model(
            os.path.join("artifacts", "training", "model.h5")
        )

        # Load & resize image
        img = image.load_img(self.filename, target_size=(224, 224))
        img_array = image.img_to_array(img)

        # Expand batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        # ResNet0 preprocessing
        img_array = preprocess_input(img_array)

        # Predict
        predictions = model.predict(img_array)

        # Output
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions)

        logger.info(f"Predictions: {predictions}")


        label = []


        if predicted_class == 0:
            prediction = "bengin"
            return prediction

        elif predicted_class == 1:
            prediction ="malignant"
            return prediction
        else:
            prediction = "normal"
            return prediction
        label = label.append(prediction)
    
        logger.info(f"label: {label}")
