import os 
import numpy as np
import tensorflow as tf
from PIL import Image
from urllib import request
from io import BytesIO
from tensorflow.keras.models import load_model

# model = load_model('bees-wasps.h5', compile=False)

# optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
# model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# tflite_model = converter.convert()

# with open('converted_model.tflite', 'wb') as f:
#     f.write(tflite_model)
    

MODEL_NAME = os.path.abspath('converted_model.tflite')

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def prepare_input(x): 
    return x / 255.0

interpreter = tf.lite.Interpreter(model_path=MODEL_NAME)
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index'] 
output_index = interpreter.get_output_details()[0]['index']

def predict_url(url): 
    img = download_image(url)
    img = prepare_image(img, target_size=(150, 150))
    x = np.array(img, dtype='float32')
    X = np.array([x])
    X = prepare_input(X)
    
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)
    
    return float(preds[0, 0])

def lambda_handler(event, context):
    url = event['url']
    pred = predict_url(url) 
    result = {
        'prediction': pred
    }
    return result 