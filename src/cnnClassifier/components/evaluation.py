import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
import dagshub

from cnnClassifier.utils.common import save_json
from cnnClassifier.entity.config_entity import  EvaluationConfig




dagshub.init(repo_owner='harcuracy', repo_name='Mlops-Production-Ready-Deep-Learning-Project', mlflow=True)

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config


    def _valid_generator(self):

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:2],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear",
            shuffle=False
        )

        #CASE 1: Use separate testing data
        if self.config.testing_data is not None:
            valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rescale=1. / 255
            )

            self.valid_generator = valid_datagenerator.flow_from_directory(
                directory=self.config.testing_data,
                **dataflow_kwargs
            )

        #CASE 2: Use validation split from training data
        else:
            valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rescale=1. / 255,
                validation_split=0.30
            )

            self.valid_generator = valid_datagenerator.flow_from_directory(
                directory=self.config.training_data,
                subset="validation",
                **dataflow_kwargs
            )



    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)



    def evaluate(self):
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)


    def save_score(self):
        scores = {"loss":self.score[0],"accuracy":self.score[1]}
        save_json(path= Path("scores.json"),data= scores)


    def log_into_mlflow(self):
        # Set DagsHub as tracking and registry
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        mlflow.set_registry_uri(self.config.mlflow_uri)
        
        #Set experiment
        mlflow.set_experiment("chest cancer experiment")
        
        with mlflow.start_run():
            # log dataset type
            mlflow.log_param(
                "evaluation_dataset",
                "testing" if self.config.testing_data else "validation_split"
            )
            
            # log params and metrics
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics({
                "loss": self.score[0],
                "accuracy": self.score[1]
            })
            
            # log model with a name
            mlflow.keras.log_model(
                self.model,
                "model",
                registered_model_name="resnet50Model"
            )


    

    
        
                                                                           
