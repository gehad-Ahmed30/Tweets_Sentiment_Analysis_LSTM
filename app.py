from flask import Flask, render_template, request, jsonify
import os
import pickle
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.initializers import Orthogonal


nltk.download('stopwords')
stop_words = set(stopwords.words('english')) - {'not', 'no', 'never', 'nothing', 'nowhere'}

current_dir = os.path.dirname(os.path.abspath(__file__))



model = load_model('BestLstm_Model.h5',custom_objects={'Orthogonal': Orthogonal})

# ---- Load Tokenizer & LabelEncoder ----
with open('tokenizer_acc.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

with open('label_encoder_acc.pkl', 'rb') as f:
    label_encoder = pickle.load(f)


max_len = 80

def clean_text(text):
    if not isinstance(text, str):
        text = ""
    text = text.lower()
    text = re.sub(r'@[\w]+', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), 'templates'))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    user_input = request.form.get("tweet", "")
    if not user_input.strip():
        prediction = "Please enter a valid tweet."
    else:
        cleaned = clean_text(user_input)
        seq = tokenizer.texts_to_sequences([cleaned])
        pad_seq = pad_sequences(seq, maxlen=max_len, padding='post')
        pred = model.predict(pad_seq, verbose=0)
        label_index = np.argmax(pred, axis=1)
        prediction = label_encoder.inverse_transform(label_index)[0]

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)


