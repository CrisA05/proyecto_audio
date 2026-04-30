import speech_recognition as sr
import threading
from spellchecker import SpellChecker
import re

from SVM2 import SVMInputs

spell = SpellChecker(language='en')
file_path = 'ejemplo.txt'

#diccionario
spell.word_frequency.load_words([])

def corregir_texto_basico(texto):
    palabras = re.findall(r'\w+|\S', texto)

    corregidas = []
    for palabra in palabras:
        if palabra.isalpha():
            if palabra.lower() in spell:
                corregidas.append(palabra)
            else:
                sugerida = spell.correction(palabra)
                corregidas.append(sugerida if sugerida else palabra)
        else:
            corregidas.append(palabra)

    texto = " ".join(corregidas)

    texto = texto.strip()

    
    if texto:
        texto = texto[0].upper() + texto[1:]

    
    if texto and texto[-1] not in ".!?":
        texto += "."

    return texto


def leer_input():
    while True:
        texto = input()
        print(texto)

thread = threading.Thread(target=leer_input, daemon=True)
thread.start()

recognizer = sr.Recognizer()

with sr.Microphone() as source:
    recognizer.adjust_for_ambient_noise(source, duration=1)

    while True:    
        try:
            print("input: ")
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            
            text = recognizer.recognize_google(audio, language='es-MX')

            # text = corregir_texto_basico(text)

            with open(file_path, 'a', encoding='utf-8') as file:
                file.write(text + "\n")
            print(text)
            
        except sr.UnknownValueError:
            print("no se pudo entender")
        except sr.RequestError as e:
            print(f"Error: {e}")
        
        # my_inputs = SVM2.load_transcription_to_list(file_path)

        # SVM2.SVMInputs(my_inputs)