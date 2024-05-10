import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify
#change the path according to your laptop#
model_path = r'C:\Users\layan\OneDrive\Desktop\soil_photos_\Image_classify.h5' #change the path#
model = load_model(model_path)

app = Flask(__name__ , template_folder='templates', static_folder='static', static_url_path='/')

def predict(image):
    image_array = tf.keras.utils.img_to_array(image)
    image_batch = tf.expand_dims(image_array, axis=0)
    prediction = model.predict(image_batch)
    data_cat = ['Gravel AND The right plant to plant in this soil is Cistus', 
                'Sand AND The right plant to plant in this soil is Tulips', 
                'Silt AND The right plant to plant in this soil is Japanese iris']
    result = data_cat[prediction.argmax()]
    accuracy = prediction.max() * 100
    return result, accuracy

def is_valid_image(file):
    allowed_extensions = {'png', 'jpg', 'jpeg'}
    return '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in allowed_extensions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_image():
    if 'file' not in request.files:
        return 'No file uploaded'
    
    image_file = request.files['file']

    if not is_valid_image(image_file):
        return 'Invalid file format. Please upload a PNG or JPEG image.'

    try:
        image = Image.open(image_file).resize((180, 180))
        result, accuracy = predict(image)

        if accuracy < 15:
            return 'The uploaded photo does not relate to soil with sufficient accuracy.'

        return result
    except:
        return 'Error processing the image'

if __name__ == '__main__':
    app.run(debug=True)
