import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import random
import pickle
import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    with open(os.path.join(PROJECT_DIR, 'data.json'), 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"Error: data.json not found in {PROJECT_DIR}.")
    exit()

try:
    nltk.download("punkt")
    nltk.download("wordnet")
    nltk.download('punkt_tab')
except LookupError:
    pass

lemmatizer = WordNetLemmatizer()

words = []
classes = []
documents = []
ignore_words = ["?", "!"]

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        word_list = word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(set(words))

classes = sorted(set(classes))

pickle.dump(words, open(os.path.join(PROJECT_DIR, "words.pkl"), "wb"))
pickle.dump(classes, open(os.path.join(PROJECT_DIR, "classes.pkl"), "wb"))

training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    word_patterns = doc[0]
    word_patterns = [lemmatizer.lemmatize(w.lower()) for w in word_patterns]
    for w in words:
        bag.append(1) if w in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)

train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation="softmax"))

sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

model.save(os.path.join(PROJECT_DIR, "chatbot_model.h5"))

print("Training complete. Model saved.")