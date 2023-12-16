from flask import Flask, request, jsonify
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app,origin=['*','http://localhost:3000'],allow_headers=['Content-Type','Authorization','Access-Control-Allow-Credentials','Access-Control-Allow-Origin','Access-Control-Allow-Headers','x-xsrf-token','Access-Control-Allow-Methods','Access-Control-Allow-Headers','Access-Control-Allow-Headers','Access-Control-Allow-Origin','Access-Control-Allow-Methods','Authorization','X-Requested-With','Access-Control-Request-Headers','Access-Control-Request-Method'])
port = 4000

# Load the model
model = load_model("keras_model.h5", compile=False)

# Load the labels
class_names = [line.strip() for line in open("labels.txt", "r").readlines()]

# Confidence threshold for predictions
confidence_threshold = 0.7

@app.route('/')
def hello():
    return f"Server is running on port {port}"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # Process the image file
        image = Image.open(file.stream).convert("RGB")
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

        # Convert image to numpy array and normalize
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

        # Prepare the image for prediction
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
            # Classify as "other" if confidence is below threshold
            class_name = "Other"
            confidence_score = float(max_confidence)

        # Return the result
        response = jsonify({'class': class_name, 'confidence_score': confidence_score})
    return response
if __name__ == '__main__':
    app.run(debug=True, port=port)
