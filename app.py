import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from flask import Flask,request,app,jsonify,url_for,render_template

app=Flask(__name__)

model = tf.keras.models.load_model("urdu_all_ner_model_train.h5")
word_model = Word2Vec.load('urdu_all_word2vec.bin')
idx2label = {0: 'Location', 1:  'Organization', 2:'Other', 3:'Person'}

def vectorize_data(data, vocab: dict) -> list:
    keys = list(vocab.keys())
    filter_unknown = lambda word: vocab.get(word, None) is not None
    encode = lambda review: list(map(keys.index, filter(filter_unknown, review)))
    vectorized = list(map(encode, data))
    return vectorized


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods =['POST'])
def predict__api():
    data = request.form['data']
    token = word_tokenize(data)
    pad = pad_sequences(
        sequences=vectorize_data(token, vocab=word_model.wv.key_to_index),
        maxlen=50,
        padding='post')
    output = model.predict(pad)
    output = np.argmax(output, axis=1)
    prediction = [int(val) for val in output]
    pred_tag_list = [idx2label[tag_id] for tag_id in prediction]
    result = [{"word": word, "tag": tag} for word, tag in zip(data.split(), pred_tag_list) if tag != 'O']
    result_text = " ".join([item['word'] for item in result])
    return render_template('result.html',result_text=data, result=result)

if __name__ == "__main__":
    app.run(debug=True, port=5001)
    