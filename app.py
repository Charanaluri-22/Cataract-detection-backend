from flask import Flask, request, jsonify
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
from flask_cors import CORS
import cv2
import os
from dotenv import load_dotenv

 # Load environment variables from .env and .flaskenv
load_dotenv()
# Load Haar cascade classifiers for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

frontend_url = os.getenv('FRONTEND_URL', 'http://localhost:3000')

app = Flask(__name__)
CORS(app,origin=[frontend_url],allow_headers=['Content-Type','Authorization','Access-Control-Allow-Credentials','Access-Control-Allow-Origin','Access-Control-Allow-Headers','x-xsrf-token','Access-Control-Allow-Methods','Access-Control-Allow-Headers','Access-Control-Allow-Headers','Access-Control-Allow-Origin','Access-Control-Allow-Methods','Authorization','X-Requested-With','Access-Control-Request-Headers','Access-Control-Request-Method'])
port = 5000

# Load the model
model = load_model("keras_model.h5", compile=False)

# Load the labels
class_names = [line.strip() for line in open("labels.txt", "r").readlines()]

# Confidence threshold for predictions
confidence_threshold = 0.6

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Convert uploaded image to OpenCV format
    file_bytes = np.frombuffer(file.read(), np.uint8)
    cv_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    eye_img = None

    # Try detecting eyes directly in the full image
    eyes = eye_cascade.detectMultiScale(gray)
    if len(eyes) > 0:
        ex, ey, ew, eh = eyes[0]
        eye_img = cv_img[ey:ey + eh, ex:ex + ew]
    else:
        # Try detecting face and extract eye
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = cv_img[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            if len(eyes) > 0:
                ex, ey, ew, eh = eyes[0]
                eye_img = roi_color[ey:ey + eh, ex:ex + ew]
                break

    # If no eye image was found
    if eye_img is None:
        return jsonify({'error': 'Invalid image: No face or eye detected'}), 400

    # Resize and preprocess the eye image
    eye_img_rgb = cv2.cvtColor(eye_img, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(eye_img_rgb).convert("RGB")
    size = (224, 224)
    image_pil = ImageOps.fit(image_pil, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image_pil)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Predict
    prediction = model.predict(data)
    max_confidence = np.max(prediction)

    if max_confidence >= confidence_threshold:
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = float(prediction[0][index])
    else:
        class_name = "Other"
        confidence_score = float(max_confidence)

    return jsonify({'class': class_name, 'confidence_score': confidence_score})
@app.route("/", methods=['GET'])
def index():
    return jsonify({'message': 'Welcome to the Eye Detection API!'}), 200

if __name__ == '__main__':
    app.run(debug=True, port=port)
