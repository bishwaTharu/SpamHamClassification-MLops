from sklearn.linear_model import LogisticRegression

class SpamHamModel:
    def __init__(self, random_state: int):
        self.model = LogisticRegression(random_state=random_state)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
