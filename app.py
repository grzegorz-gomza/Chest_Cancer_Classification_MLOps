from flask import Flask, jsonify, render_template, request, redirect, url_for
from ChestCancerClassifier.utils.model_handler import ModelHandler
import os
from werkzeug.utils import secure_filename
from ChestCancerClassifier.config.configuration import ConfigurationManager

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
    return "." in filename and filename.rsplit(".",1)[1].lower() in ALLOWED_EXTENTIONS

@app.route('/')
def index():
    sample_images = os.listdir(SAMPLE_FOLDER)
    return render_template('index.html', 
                         sample_images=sample_images)

@app.route('/predict', methods=['POST'])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No file provided"})

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error: No file selected"})

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        prediction = model_handler.predict(filepath)
        return jsonify({"prediction": prediction})

    return jsonify({"error": "Invalid file type"})

@app.route('/predict_sample', methods=['POST'])
def predict_sample():
    filename = request.form.get("sample_image")
    if not filename:
        return jsonify({"error": "No filename provided"})

    filepath = os.path.join(SAMPLE_FOLDER, filename)
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"})
    prediction = model_handler.predict(filepath)
    return jsonify({"prediction": prediction})

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)