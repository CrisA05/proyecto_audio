import re
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold
from nltk.corpus import stopwords
import kagglehub
import os
import joblib
from joblib import Parallel, delayed

url = "https://huggingface.co/datasets/thesofakillers/jigsaw-toxic-comment-classification-challenge/resolve/main/train.csv"

df = pd.read_csv(url)
print(df.columns.tolist())

df = df.drop('id', axis=1)

label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
df['Label'] = df[label_cols].max(axis=1)

CONTRACTIONS = {
    "what's": "what is", "don't": "do not", "aren't": "are not", "isn't": "is not",
    "that's": "that is", "doesn't": "does not", "he's": "he is", "she's": "she is",
    "it's": "it is", "i'm": "i am", "e-mail": "email", "u s ": " american ", " us ": " american "
}

STOPWORDS = set(stopwords.words('english'))
STEMMER = SnowballStemmer("english", ignore_stopwords=True)
LEMMATIZER = WordNetLemmatizer()

def my_clean(text, stops=False, stemming=False):
    text = str(text).lower()

    for word, replacement in CONTRACTIONS.items():
        text = text.replace(word, replacement)

    text = re.sub(r"[^a-z0-9\s]", " ", text)

    text = re.sub(r"(\d+)k", r"\1000", text)
    words = text.split()

    words = [w for w in words if len(w) >= 2]

    if stops:
        words = [w for w in words if w not in STOPWORDS]

    if stemming:
        words = [LEMMATIZER.lemmatize(STEMMER.stem(w)) for w in words]
        if stops:
            words = [w for w in words if w not in STOPWORDS]

    return " ".join(words)


class Preproccesor:
    def __init__(self):
        pass

    @staticmethod
    def load_data(df_input, preprocessed=True, stemming_a=True):
        data = df_input.sample(frac=1, random_state=2000).reset_index(drop=True)

        y = (data['Label'].values == 1).astype(int)

        XT = data['Content'].values

        if preprocessed:
            X = Parallel(n_jobs=2, batch_size=1000)(
                delayed(my_clean)(str(x), stops=False, stemming=stemming_a)
                for x in XT
            )
        else:
            X = XT

        return np.array(X), y
    
N_SAMPLES = 32000

df_limited = (
    df.groupby('Label', group_keys=False)
    .apply(lambda g: g.sample(n=min(len(g), N_SAMPLES // 2), random_state=42))
    .reset_index(drop=True)
)

df_limited.rename(columns={'comment_text': 'Content'}, inplace=True)

X, y = Preproccesor.load_data(df_limited, True)

f1ethos = []
f1ethosH = []
f1ethosNH = []
accethos = []

best_f1 = 0
best_svm = None
best_vec = None

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for train_index, test_index in kf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    if len(np.unique(y_train)) < 2:
        print("Skipping fold: only one class in training set")
        continue

    svm = LinearSVC(dual=False, max_iter=2000, class_weight='balanced')

    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=30000)
    X_tr = vec.fit_transform(X_train)
    X_te = vec.transform(X_test)
    svm.fit(X_tr, y_train)

    y_predict = svm.predict(X_te)
    current_f1 = f1_score(y_test, y_predict, average='weighted')

    if current_f1 > best_f1:
        best_f1 = current_f1
        best_svm = svm
        best_vec = vec

    accethos.append(balanced_accuracy_score(y_test, y_predict))
    f1ethos.append(current_f1)
    f1ethosNH.append(f1_score(y_test, y_predict, average=None)[0])
    f1ethosH.append(f1_score(y_test, y_predict, average=None)[1])

def SVMInputs(inputs):

  for input in inputs:
    input_clean = my_clean(text=str(input), stops=False, stemming=False)
    vector = best_vec.transform([input_clean])
    prediction = best_svm.predict(vector)

    print(f'Input: {input}. Prediction: {prediction[0]}.')

inputs = ["Women deserve to be treated equally", "Love is the key", "You all fucking suck", "i hate christians so much", "jews are the trash of this world"]

import pickle

# Save
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(best_vec, f)

# Load
with open('vectorizer.pkl', 'rb') as f:
    loaded_vectorizer = pickle.load(f)

# Save
with open('model.pkl', 'wb') as f:
    pickle.dump(best_svm, f)

# Load
with open('model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

def load_transcription_to_list(filepath):
    transcription_inputs = []

    # Encodings to try in order of preference
    encodings = ['utf-8', 'latin-1', 'cp1252']

    for encoding in encodings:
        try:
            with open(filepath, 'r', encoding=encoding) as file:
                for line in file:
                    line_text = line.strip()
                    if line_text:
                        transcription_inputs.append(line_text)

            return transcription_inputs

        except UnicodeDecodeError:
            # If utf-8 fails, quietly catch the error and try the next encoding in the list
            continue

        except FileNotFoundError:
            print(f"Error: Could not find the file at '{filepath}'.")
            return []
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return []

    # If the loop finishes without returning, all encodings failed
    print(f"Error: Could not decode '{filepath}' with any standard text encodings.")
    return []