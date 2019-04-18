from flask import Flask
from flask import request
import json
from classifier import ReviewClassifier
app = Flask(__name__)

classifier = ReviewClassifier()


@app.route('/classify', methods=(["POST"]))
def classify():
    text_json = request.get_json()
    text = text_json['text']
    return classifier.classify_text(text)