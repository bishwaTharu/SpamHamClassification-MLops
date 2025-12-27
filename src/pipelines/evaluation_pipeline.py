from sklearn.metrics import f1_score
from src.config.settings import Settings
from src.registry.model_registry import ModelPromoter
from src.utils.logger import get_logger

logger = get_logger(__name__)

class EvaluationPipeline:
    def __init__(self):
        self.promoter = ModelPromoter()

    def evaluate_and_promote(self, y_true, y_pred):
        f1 = f1_score(y_true, y_pred, pos_label='spam')
        logger.info(f"Evaluation F1 score: {f1}")

        self.promoter.promote_if_valid(f1)
        logger.info("Model promoted to STAGING")

        return f1
