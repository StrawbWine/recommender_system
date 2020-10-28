import numpy as np
import pandas as pd
import pickle

from flask import Flask, request, jsonify, render_template
from waitress import serve

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('./index.html')

@app.route('/predict', methods=['POST'])
def predict():
    return render_template('./index.html',
                            prediction_text='0')

if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=8080)