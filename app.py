### Main APP ###

# import os
# import json
# from flask import Flask, render_template, request, jsonify, url_for
# from keras.models import load_model
# from keras.preprocessing import image
# import numpy as np
# from werkzeug.utils import secure_filename

# app = Flask(__name__)

# # Load the pre-trained InceptionResNetV2 model


# # Set the upload folder
# UPLOAD_FOLDER = 'static/uploads'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # Ensure the upload folder exists
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# # Function to load and preprocess a single image
# def preprocess_single_image(image_path):
#     img = image.load_img(image_path, target_size=(299, 299))
#     img = img.resize((299, 299))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     return img_array

# # Function to predict whether an image is real or fake
# def predict_single_image(model, image_path):
#     img_array = preprocess_single_image(image_path)
#     img_array /= 255.0  # Normalize pixel values
#     prediction = model.predict(img_array)
#     return prediction[0][0]

# # Route to render upload form
# @app.route('/')
# def index():
#     return render_template('index1.html')

# # Route to handle file upload and predict
# @app.route('/predict', methods=['GET', 'POST'])
# def predict():
#     if request.method == 'POST':
#         # Check if the post request has the file part
#         if 'file' not in request.files:
#             return render_template('index1.html', message='No file part')
#         file = request.files['file']
#         if file.filename == '':
#             return render_template('index1.html', message='No selected file')
#         if file:
#             # Save the uploaded file
#             filename = secure_filename(file.filename)
#             file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
#             # Open the image and rescale it
#           # Resize the image
#             file.save(file_path)
#             model = load_model('deepfake_model.h5')
#             # Get prediction result
#             prediction = predict_single_image(model, file_path)

#             # Generate the URL for the saved file
#             img_url = url_for('static', filename=f'uploads/{filename}')
            
#             probability = prediction
#             if probability > 0.5:
#                 result = "Deepfake Image"
#             else:
#                 result = "Real Image"
            

#             # Return the prediction result and image URL
#             response = {
#                'img_url': img_url,
#                'prediction': np.float64(prediction).item(),
#                'file_path': file_path,
#                'result': result
#             }
            
            
#             #return jsonify({'img_url': img_url, 'prediction': np.float64(prediction).item()})
#             # Optionally remove the file after prediction
            

#             return jsonify(response)

# if __name__ == '__main__':
#     app.run(debug=True, use_reloader=False)


import os
from flask import Flask, render_template, request, jsonify, url_for
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the pre-trained InceptionResNetV2 model
model = load_model('deepfake_model.h5')

# Set the upload folder
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Function to load and preprocess a single image
def preprocess_single_image(image_path):
    img = image.load_img(image_path, target_size=(299, 299))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0  # Normalize pixel values
    return img

# Function to predict whether an image is real or fake
def predict_single_image(model, image_path):
    img_array = preprocess_single_image(image_path)
    prediction = model.predict(img_array)
    return prediction[0][0]

# Route to render upload form
@app.route('/')
def index():
    return render_template('index1.html')

# Route to handle file upload and predict
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if the post request has the file part
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if file:
            # Save the uploaded file
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Load the model
            # model = load_model('deepfake_model.h5')  # Only if you want to load the model here instead of globally
            
            # Get prediction result
            prediction = predict_single_image(model, file_path)
            probability = prediction
            
            result = "Deepfake Image" if probability > 0.5 else "Real Image"
            
            # Generate the URL for the saved file
            img_url = url_for('static', filename=f'uploads/{filename}')
            
            # Return the prediction result and image URL
            response = {
                'img_url': img_url,
                'prediction': np.float64(prediction).item(),
                'result': result
            }
            
            return jsonify(response)
        
    except Exception as e:
        # Log the error and return a JSON error response
        print(f"Error: {e}")
        return jsonify({'error': 'An error occurred during prediction'}), 500

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
