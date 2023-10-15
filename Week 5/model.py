from flask import Flask, request, jsonify
import pickle
import numpy as np
from waitress import serve

app = Flask(__name__)

# Load the logistic regression model
with open('model1.bin', 'rb') as model_file:
    logistic_regression_model = pickle.load(model_file)

# Load the DictVectorizer
with open('dv.bin', 'rb') as dv_file:
    dict_vectorizer = pickle.load(dv_file)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        client_data_transformed = dict_vectorizer.transform([data])
        probability = logistic_regression_model.predict_proba(client_data_transformed)
        credit_approval_probability = probability[0][1]
        return jsonify({'probability': credit_approval_probability})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=8081)
