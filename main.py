from src.cnnClassifier import logger
from src.cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelPipeline
from src.cnnClassifier.pipeline.stage_03_training import TrainingPipeline
from src.cnnClassifier.pipeline.stage_04_evaluation import EvaluationPipeline

"""STAGE_NAME = "Data Ingestion Stage"

if __name__ == "__main__":
    try:

        logger.info(f">>>>>>>> Stage {STAGE_NAME} have started! <<<<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>>>>> Stage {STAGE_NAME} have completed! <<<<<<<<")

    except Exception as e:
        raise e
    """

STAGE_NAME = "Prepare Base Model Stage"


if __name__ == "__main__":
    try:

        logger.info(f">>>>>>>> Stage {STAGE_NAME} have started! <<<<<<<<")
        obj = PrepareBaseModelPipeline()
        obj.main()
        logger.info(f">>>>>>>> Stage {STAGE_NAME} have completed! <<<<<<<<")

    except Exception as e:
        raise e




STAGE_NAME = "Training Stage"


if __name__ == "__main__":
    try:

        logger.info(f">>>>>>>> Stage {STAGE_NAME} have started! <<<<<<<<")
        obj = TrainingPipeline()
        obj.main()
        logger.info(f">>>>>>>> Stage {STAGE_NAME} have completed! <<<<<<<<")

    except Exception as e:
        raise e
    

STAGE_NAME = "Evaluation Stage"


if __name__ == "__main__":
    try:

        logger.info(f">>>>>>>> Stage {STAGE_NAME} have started! <<<<<<<<")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f">>>>>>>> Stage {STAGE_NAME} have completed! <<<<<<<<")

    except Exception as e:
        raise e
