import os

from cnnClassifier.constants import *
from cnnClassifier.utils.common import read_yaml,create_directories
from cnnClassifier.entity.config_entity import (DataIngestionConfig,
                                                PrepareBaseModelConfig,
                                                TrainingConfig)



class ConfigurationManager:
    def __init__(self,config_file_path = CONFIG_FILE_PATH, params_file_path = PARAMS_FILE_PATH):
        self.Config = read_yaml(config_file_path)
        self.Params = read_yaml(params_file_path)
         
        create_directories([self.Config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.Config.data_ingestion

        create_directories([config.root_dir])
        

        data_ingestion_config = DataIngestionConfig(
            root_dir = config.root_dir,
            source_URL = config.source_URL,
            local_data_file = config.local_data_file,
            unzip_dir = config.unzip_dir
        )

        return data_ingestion_config
    

    def get_prepare_base_model_config(self) ->PrepareBaseModelConfig:
        config = self.Config.prepare_base_model

        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir= config.root_dir,
            base_model_path = Path(config.base_model_path),
            updated_base_model_path = Path(config.updated_base_model_path),
            params_image_size = self.Params.IMAGE_SIZE,
            params_learning_rate = float(self.Params.LEARNING_RATE),
            params_include_top = self.Params.INCLUDE_TOP,
            params_weights = self.Params.WEIGHTS,
            params_classes = self.Params.CLASSES
              )
        return prepare_base_model_config
    
    def get_training_config(self) -> TrainingConfig:
        training = self.Config.training
        prepare_base_model = self.Config.prepare_base_model
        params = self.Params
        training_data = os.path.join(self.Config.data_ingestion.unzip_dir,"Data","train")
        validation_data = os.path.join(self.Config.data_ingestion.unzip_dir,"Data","valid")

        create_directories([Path(training.root_dir)])

        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            updated_base_model_path= Path(prepare_base_model.updated_base_model_path),
            training_data= Path(training_data),
            validation_data= None , # Path(validation_data),
            params_epochs= params.EPOCHS,
            params_batch_size= params.BATCH_SIZE,
            params_image_size= params.IMAGE_SIZE,
            params_is_augmentation= params.AUGMENTATION,

  )
        return training_config


        
        
   
