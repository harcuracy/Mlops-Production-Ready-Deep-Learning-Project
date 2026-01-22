from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.evaluation import Evaluation
from cnnClassifier import logger



STAGE_NAME = "Evaluation Stage"



class EvaluationPipeline:
    def __init__(self):
        pass
    def main(self):
        config = ConfigurationManager()
        evaluation_config = config.get_evaluation_config()
        evaluation = Evaluation(config = evaluation_config)
        evaluation.evaluate()
        evaluation.save_score()
        #evaluation.log_into_mlflow()


if __name__ == "__main__":
    try:

        logger.info(f">>>>>>>> Stage {STAGE_NAME} have started! <<<<<<<<")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f">>>>>>>> Stage {STAGE_NAME} have completed! <<<<<<<<")

    except Exception as e:
        raise e
