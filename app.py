from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import requests
from io import BytesIO
from flask_bootstrap import Bootstrap4
from ultralytics import YOLO
import cv2
import io
from skimage.io import imread
from skimage import color, transform
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu, gaussian
from skimage.transform import resize
from numpy import asarray
from skimage.metrics import structural_similarity
from skimage import measure
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import joblib
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from natsort import natsorted
from sklearn.naive_bayes import GaussianNB
from keras.models import load_model
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Ajoutez cette ligne avant d'importer matplotlib pour utiliser le backend non interactif
import matplotlib.pyplot as plt

app = Flask(__name__,static_folder='static')
bootstrap = Bootstrap4(app)

@app.route('/')
def hello():
    return render_template('index.html', image_url='/static/m.jpg')

uploaded_image = None

@app.route('/verifier', methods=['POST'])
def verifier():
    global uploaded_image
    success_message = ''
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file:
            uploaded_image = imread(uploaded_file)
            uploaded_image_gray = color.rgb2gray(uploaded_image)
            uploaded_image_resized = transform.resize(uploaded_image_gray, (1572, 2213))

            # Read and preprocess the reference images
            def preprocess_image(filepath):
                image = imread(filepath)
                image_gray = color.rgb2gray(image)
                return transform.resize(image_gray, (1572, 2213))

            image1_resized = preprocess_image('static/PMI(1).jpg')
            image2_resized = preprocess_image('static/HB(1).jpg')
            image3_resized = preprocess_image('static/Normal(1).jpg')
            image4_resized = preprocess_image('static/MI(1).jpg')

            def detect_objects(image):
                model = YOLO('best (2).pt')
                detections = model(image)
                return detections

            similarity_score = max(
                structural_similarity(uploaded_image_resized, image1_resized, data_range=image1_resized.max()),
                structural_similarity(uploaded_image_resized, image2_resized, data_range=image2_resized.max()),
                structural_similarity(uploaded_image_resized, image3_resized, data_range=image3_resized.max()),
                structural_similarity(uploaded_image_resized, image4_resized, data_range=image4_resized.max())
            )

            try:
                detections = detect_objects(uploaded_image)
                confidences = detections[0].boxes.conf 
                if confidences is not None and any(conf >= 0.7 for conf in confidences):
                    success_message = "This is an ECG image."
                elif similarity_score < 0.70:
                    success_message = "This is not an ECG image."
                else:
                    success_message = "Uncertain result."
            except Exception as e:
                success_message = f"Error processing the image: {str(e)}"

    return render_template('index.html', success_message=success_message)




@app.route('/result', methods=['GET', 'POST'])
def result():
    try:
        if request.method == 'POST':
            Lead_1 = uploaded_image[300:600, 150:643]
            Lead_2 = uploaded_image[300:600, 646:1135]
            Lead_3 = uploaded_image[300:600, 1140:1625]
            Lead_4 = uploaded_image[300:600, 1630:2125]
            Lead_5 = uploaded_image[600:900, 150:643]
            Lead_6 = uploaded_image[600:900, 646:1135]
            Lead_7 = uploaded_image[600:900, 1140:1625]
            Lead_8 = uploaded_image[600:900, 1630:2125]
            Lead_9 = uploaded_image[900:1200, 150:643]
            Lead_10 = uploaded_image[900:1200, 646:1135]
            Lead_11 = uploaded_image[900:1200, 1140:1625]
            Lead_12 = uploaded_image[900:1200, 1630:2125]
            Lead_13 = uploaded_image[1250:1480, 150:2125]
            Leads=[Lead_1,Lead_2,Lead_3,Lead_4,Lead_5,Lead_6,Lead_7,Lead_8,Lead_9,Lead_10,Lead_11,Lead_12,Lead_13]
            

            fig, ax = plt.subplots(4, 3)
            fig.set_size_inches(15, 15)
            for i, lead in enumerate(Leads[:-1]):
                ax[i // 3, i % 3].imshow(lead)
                ax[i // 3, i % 3].axis('off')
                ax[i // 3, i % 3].set_title(f"Leads {i+1}")
            plt.savefig('static/Leads_1-12_figure.png', dpi=300)

          
            leads = f"static/Leads_1-12_figure.png"

           

            return render_template('result.html', leads=leads, prediction=None)
    except Exception as e:
        print(f"Error: {e}")
        return f"Error processing the result: {e}", 500

def predict_disease(image_path):
    # Read and resize the input image using Pillow (PIL)
    with Image.open(image_path) as img:
        img = img.resize((255, 255))
        img = np.array(img)

    # Perform prediction on the image
    results = model(img, show=True)

    # Extract relevant information from the prediction
    names_dict = results[0].names
    probs = results[0].probs.data.tolist() 
    
    prediction = names_dict[probs.index(max(probs))]

    return prediction

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the request
    file = request.files['file']

    # Save the file temporarily
    image_path = 'uploaded_image.jpg'
    file.save(image_path)

    # Pass the image to the YOLO model for prediction
    prediction = predict_disease(image_path)

    # Render the template with the prediction result
    return render_template('result.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)


