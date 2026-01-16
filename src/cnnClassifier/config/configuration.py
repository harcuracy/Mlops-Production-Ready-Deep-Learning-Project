from cnnClassifier.constants import *
from cnnClassifier.utils.common import read_yaml,create_directories
from cnnClassifier.entity.config_entity import (DataIngestionConfig)



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


        
        
   
