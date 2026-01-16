from src.cnnClassifier import logger
from src.cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline



STAGE_NAME = "Data Ingestion Stage"


if __name__ == "__main__":
    try:

        logger.info(f">>>>>>>> Stage {STAGE_NAME} have started! <<<<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>>>>> Stage {STAGE_NAME} have completed! <<<<<<<<")

    except Exception as e:
        raise e