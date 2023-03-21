import os
import urllib.request
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename
import onnxruntime as ort
import numpy as np
from PIL import Image

# Set up the Flask app
app = Flask(__name__)
#app.secret_key = "super_secret_key"
app.config['UPLOAD_FOLDER'] = 'uploads'


# Set up the ONNX model
model_path = 'googlenet-12-int8.onnx'
session = ort.InferenceSession(model_path)
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape

# Define the allowed file extensions for image uploads
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Function to check if a file has an allowed extension
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Pre-processing function for ImageNet models using numpy
def preprocess(img):
    # Resize the image to the model's input shape
    img = np.array(Image.fromarray(img).resize((input_shape[3], input_shape[2]))).astype(np.float32)
    
    # Subtract the mean RGB values and convert from HWC to CHW format
    img[:, :, 0] -= 123.68
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 103.939
    img = img.transpose((2, 0, 1))
    
    # Remove any dimensions with size 1
    img = np.squeeze(img)
    
    # Add a batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img

# Function to predict the class of an image
def predict(img):
    img_np = np.asarray(img)
    input_data = preprocess(img_np)
    output = session.run(None, {input_name: input_data})
    prob = output[0].squeeze()
    class_labels = np.loadtxt('synset_words.txt', str, delimiter='\t')
    top_k = np.argsort(prob)[::-1][:5]
    results = []
    for i in top_k:
        results.append((class_labels[i], round(float(prob[i]), 2)))
    return results

# Define the route for the home page
@app.route('/')
def home():
    return render_template('home.html')

# Define the route for image uploads
@app.route('/predict', methods=['POST'])
def upload_file():
    # Check if a file was submitted
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    # Get the uploaded file
    file = request.files['file']
    
    # Check if the file has an allowed extension
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if not allowed_file(file.filename):
        flash('Invalid file type. Please upload an image file.')
        return redirect(request.url)
    
    # Save the file to a directory on the server
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    
    # Load the image and predict its class
    img = Image.open(file.stream)
    results = predict(img)
    top_result = results[0]
    
    # Render the results page with the image and predicted class
    return render_template('results.html', image_name=filename, class_label=top_result[0], probability=top_result[1], results=results)


if __name__ == '__main__':
    app.run(debug=True)