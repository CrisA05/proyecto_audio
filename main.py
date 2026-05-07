import speech_recognition as sr
from modelo import ToxicityModel

model = ToxicityModel("model.pkl", "vectorizer.pkl")

recognizer = sr.Recognizer()

print("Habla... (Ctrl+C para salir)")

while True:
    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5) #duracion en segundos
            audio = recognizer.listen(source)

        texto = recognizer.recognize_google(audio, language="en")
        print(f"\nTexto: {texto}")

        pred, prob = model.predict(texto)

        print(f"Clasificación: {pred} (confianza aprox: {prob:.2f})")

    except sr.UnknownValueError:
        print("No se entendió el audio")
    except sr.RequestError as e:
        print(f"Error con el servicio: {e}")
    except KeyboardInterrupt:
        print("\nSaliendo...")
        break


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

        # Probabilidad 
        if hasattr(self.model, "decision_function"):
            score = self.model.decision_function(X)[0]
            prob = 1 / (1 + pow(2.718, -score))
        else:
            prob = 0.0

        label = "TOXICO" if pred == 1 else "NO TOXICO"
        return label, prob
