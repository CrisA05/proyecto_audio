import pickle

class ToxicityModel:
    def __init__(self, model_path, vectorizer_path):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)

    def predict(self, text):
        X = self.vectorizer.transform([text])
        pred = self.model.predict(X)[0]

        if hasattr(self.model, "decision_function"):
            score = self.model.decision_function(X)[0]
            prob = 1 / (1 + pow(2.718, -score))
        else:
            prob = 0.0

        label = "TOXICO" if pred == 1 else "NO TOXICO"
        return label, prob