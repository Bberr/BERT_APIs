import multiprocessing
import os
import sys

from flask import Flask, jsonify, request
from flask_cors import CORS
from simpletransformers.classification import ClassificationModel, ClassificationArgs

import base64
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from io import BytesIO

root_path = os.path.dirname(os.getcwd())
sys.path.insert(0, root_path)


def myModel(text):
    BERT_model = ClassificationModel(
        "bert", "best_model",
        use_cuda=False
    )
    predictions, raw_outputs = BERT_model.predict(text)
    sentiment = ['Negative', 'Neutral', 'Positive']
    return sentiment[predictions[0]]


def myModelConfusion(text,label):
    test_data = text
    test_label = label
    labels = []
    for i in test_label:
        if i == '1':
            labels.append(0)
        if i == '3':
            labels.append(1)
        if i == '5':
            labels.append(2)

    BERT_model = ClassificationModel(
        "bert", "best_model",
        use_cuda=False
    )
    predictions, raw_outputs = BERT_model.predict(test_data)
    print(predictions)
    print(labels)
    bert_accuracy = accuracy_score(labels, predictions)

    # bert_cn = confusion_matrix(test_label, predictions)
    # plt.subplots(figsize=(10, 10))
    # sns.heatmap(bert_cn, annot=True, fmt="1d", cbar=False, xticklabels=labels, yticklabels=labels)
    # plt.title("Bert Accuracy: {}".format(bert_accuracy), fontsize=20)
    # plt.xlabel("Predicted", fontsize=15)
    # plt.ylabel("Actual", fontsize=15)
    # buffer = BytesIO()
    # plt.savefig(buffer, format='png')
    # buffer.seek(0)
    # b64_image = base64.b64encode(buffer.read()).decode('utf-8')
    #return b64_image
    return bert_accuracy


if __name__ == '__main__':
    app = Flask(__name__)
    CORS(app, origins=["http://localhost:3000"], supports_credentials=True, methods=["GET", "POST"])


    @app.route('/')
    def welcome():
        print('Hello AI rate')
        return 'Hello AI rate'


    @app.route('/rate', methods=['POST'])
    def rate():
        input_text = request.json['data']
        input_text = [input_text]
        print(input_text)
        result = myModel(input_text)
        print(myModel(input_text))
        return jsonify({
            'rate': result
        })


    @app.route('/confusionMatrix', methods=['POST'])
    def confusionMatrix():
        input_text = request.json['text']
        input_label = request.json['label']
        b64_image = myModelConfusion(input_text,input_label)
        return jsonify({
            'confusion_matrix': b64_image
        })


    multiprocessing.set_start_method('spawn')
    multiprocessing.set_executable(os.path.join(sys.executable))
    app.run(host='0.0.0.0', port=8080, debug=False)
