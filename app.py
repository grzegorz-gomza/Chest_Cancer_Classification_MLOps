# app.py
from flask import Flask, jsonify, render_template, request, redirect, url_for
from ChestCancerClassifier.utils.model_handler import ModelHandler
import os
from werkzeug.utils import secure_filename
from ChestCancerClassifier.config.configuration import ConfigurationManager
import base64
import io
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__,
            template_folder="frontend",
            static_folder="frontend")

UPLOAD_FOLDER = "frontend/uploads"
SAMPLE_FOLDER = "frontend/sample_images"
ALLOWED_EXTENTIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

config = ConfigurationManager()
evaluate_model_config = config.get_evaluation_config()
model_handler = ModelHandler(evaluate_model_config)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENTIONS

def prediction_handler(predictions):
    if not isinstance(predictions, str):
        probabilities = {'adenocarcinoma': predictions[0][0],
        'large cell carcinoma': predictions[0][1],
        'normal': predictions[0][2],
        'squamous cell carcinoma': predictions[0][3]}

        sorted_probabilities = dict(sorted(probabilities.items(), reverse=True, key=lambda x:x[1]))
        
        def check_if_cancer(prob):
            if list(prob.keys())[0] == 'normal' and list(prob.values())[0] > 0.5:
                return False
            else:
                return True
        
        is_cancer = check_if_cancer(sorted_probabilities)
        
        if not is_cancer:
            return f"Cancer not detected with probability of {sorted_probabilities['normal'] * 100:.2f}%"
        else:
            cancer_prob = 1 - sorted_probabilities['normal']
            return f"Cancer detected with probability of {cancer_prob * 100:.2f}%. The cancer type is most probably {list(sorted_probabilities.keys())[0]} with probability of {sorted_probabilities[list(sorted_probabilities.keys())[0]]/cancer_prob * 100:.2f}%"
    else:
        return predictions


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_results = "No prediction yet. Please upload or select an image and click 'Get Prediction'."
    sample_images = os.listdir(SAMPLE_FOLDER)

    if request.method == 'POST':
        if "image" in request.files and request.files["image"].filename != "":
            file = request.files["image"]
            if allowed_file(file.filename):
                image = Image.open(file.stream)
                image_np = np.array(image)
                prediction = model_handler.predict(image_np)
                prediction_results = prediction
            else:
                prediction_results = f"error: Invalid file type. Supported file types: {ALLOWED_EXTENTIONS}"
        elif "selected_sample" in request.form and request.form["selected_sample"]:
            filename = request.form["selected_sample"]
            filepath = os.path.join(SAMPLE_FOLDER, filename)
            if not os.path.exists(filepath):
                prediction_results = "error : File not found"
            else:
                image = Image.open(filepath)
                image_np = np.array(image)
                prediction = model_handler.predict(image_np)
                prediction_results = prediction
        else:
            prediction_results = "error :No image provided for prediction"

    prediction_to_display = prediction_handler(prediction_results)
    
    return render_template('index.html',
                           sample_images=sample_images,
                           prediction_to_display=prediction_to_display)

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=False)
