from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

class SpamHamPipeline:
    @staticmethod
    def build(random_state: int) -> Pipeline:
        return Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("classifier", LogisticRegression(random_state=random_state))
        ])
