import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import random
import pickle
from flask import Flask, request, jsonify, send_from_directory
import os

app = Flask(__name__)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    nltk.download("punkt")
    nltk.download("wordnet")
    nltk.download('punkt_tab')
except LookupError:
    pass

try:
    with open(os.path.join(PROJECT_DIR, 'data.json'), 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"Error: data.json not found in {PROJECT_DIR}. Create it.")
    exit()

try:
    with open(os.path.join(PROJECT_DIR, "words.pkl"), "rb") as f:
        words = pickle.load(f)
    with open(os.path.join(PROJECT_DIR, "classes.pkl"), "rb") as f:
        classes = pickle.load(f)
    model = load_model(os.path.join(PROJECT_DIR, "chatbot_model.h5"))
except FileNotFoundError:
    print(f"Error: Model files not found in {PROJECT_DIR}. Run train.py first.")
    exit()

lemmatizer = WordNetLemmatizer()

def clean_up_sentence(sentence):
    sentence_words = word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence, model):
    bow = bag_of_words(sentence, words)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]
    return return_list

def get_response(intents_list, intents_json):
    if intents_list:
        tag = intents_list[0]["intent"]
        for i in intents_json["intents"]:
            if i["tag"] == tag:
                return random.choice(i["responses"])
    return "I'm not sure I understand."

@app.route('/')
def index():
    return send_from_directory(PROJECT_DIR, 'index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    message = request.json.get('message')
    if message:
        ints = predict_class(message, model)
        res = get_response(ints, data)
        return jsonify({'response': res})
    return jsonify({'error': 'No message provided'})

if __name__ == '__main__':
    app.run(debug=True)