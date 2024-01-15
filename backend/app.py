from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from PIL import Image
import io
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import numpy as np
from keras.preprocessing import image

#### pip install opencv-python-headless

app = Flask(__name__)
CORS(app)

# Define class labels for leaf classification
class_labels_of_classification = {0: 'Apple Leaf', 1: 'Grape Leaf', 2: 'Tomato Leaf'}

# Define class labels for disease detection
class_labels_of_disease_detection = {0: 'Apple Black Rot Disease', 2: 'Healthy Apple', 1: 'Apple Scab Disease', 
                                     3: 'Grape Esca (Black Measles) Disease', 5: 'Healthy Grape', 4: 'Grape Leaf Blight Disease', 
                                     6: 'Tomato Early Blight Disease', 8: 'Healthy Tomato', 7: 'Tomato Leaf Mold Disease'}

# Define class labels for water estimation
class_labels_of_water = {0: 'Dead Leaf', 1: 'Healthy Leaf', 2: 'Unhealthy Leaf'}

# Load the classification model
model_path_classification = 'leaf_classification_model_4_0.h5'
model_classification = load_model(model_path_classification)
print(f"Model loaded successfully from {model_path_classification}")

# Load the disease detection model
model_path_disease = 'disease_detection_model_3_0.h5'
model_disease = load_model(model_path_disease)
print(f"Model loaded successfully from {model_path_disease}")

# Load the water model
model_path_water = 'water_level_model_1_0.h5'
model_water = load_model(model_path_water)
print(f"Model loaded successfully from {model_path_water}")

# Define is_black_and_white function at the beginning
def is_black_and_white(img_array):
    # Check if all values in the channel are the same
    print("function is running")
    return np.all(img_array[:,:,0] == img_array[:,:,1]) and np.all(img_array[:,:,1] == img_array[:,:,2])
    print("function is running2")

# Warning message for black and white images
warning_msg = "The image is black and white, processing may not be accurate. Please upload a colorful image."

# ====================== Leaf Classification ===========================

def preprocess_image_classification(img):
    try:
        img = img.resize((500, 500))  # Resize the image

        # Convert the image to a NumPy array
        img_array = image.img_to_array(img)

        # Check if the image has 4 channels (RGBA), and convert it to RGB
        if img_array.shape[-1] == 4:
            img_array = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGBA2RGB)

        # Check if the image is black and white using custom function
        if is_black_and_white(img_array):
            print("Warning: Preprocess: The image is black and white. Further processing may not be accurate.")
            return None  # Return None for black and white images
            
        # Expand the dimensions and normalize the pixel values
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize the pixel values
        return img_array 
        
    except Exception as e:
        raise ValueError(f"Error during image preprocessing: {str(e)}")

def predict_image_classification(img, model_classification):
    try:
        preprocessed_image = preprocess_image_classification(img)

        # Check if preprocessed_image is empty (black and white image)
        if preprocessed_image is None:
            return {"warning": warning_msg}  # Return a dictionary

        # Continue with prediction for non-black and white images
        prediction = model_classification.predict(preprocessed_image)
        class_index = np.argmax(prediction)
        predicted_class = class_labels_of_classification[class_index]
        confidence = prediction[0][class_index]
        return predicted_class, confidence
        
    except Exception as e:
        raise ValueError(f"Error during image prediction: {str(e)}")

# ========================== Disease Detection ===============================

def preprocess_image_disease(img):
    try:
        img = img.resize((256, 256))  # Resize the image

        # Convert the image to a NumPy array
        img_array = image.img_to_array(img)

        # Check if the image has 4 channels (RGBA), and convert it to RGB
        if img_array.shape[-1] == 4:
            img_array = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGBA2RGB)

        # Check if the image is black and white using custom function
        if is_black_and_white(img_array):
            print("Warning: Preprocess: The image is black and white. Further processing may not be accurate.")
            return None  # Return None for black and white images
            
        # Expand the dimensions and normalize the pixel values
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize the pixel values
        return img_array 
        
    except Exception as e:
        raise ValueError(f"Error during image preprocessing: {str(e)}")

def predict_image_disease(img, model_disease):
    try:
        preprocessed_image = preprocess_image_disease(img)

        # Check if preprocessed_image is empty (black and white image)
        if preprocessed_image is None:
            return {"warning": warning_msg}  # Return a dictionary

        # Continue with prediction for non-black and white images
        prediction = model_disease.predict(preprocessed_image)
        class_index = np.argmax(prediction)
        predicted_class = class_labels_of_disease_detection[class_index]
        confidence = prediction[0][class_index]
        return predicted_class, confidence
        
    except Exception as e:
        raise ValueError(f"Error during image prediction: {str(e)}")

# ========================== Water Estimation ===============================

def preprocess_image_water(img):
    try:
        img = img.resize((500, 500))  # Resize the image

        # Convert the image to a NumPy array
        img_array = image.img_to_array(img)

        # Check if the image has 4 channels (RGBA), and convert it to RGB
        if img_array.shape[-1] == 4:
            img_array = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGBA2RGB)

        # Check if the image is black and white using custom function
        if is_black_and_white(img_array):
            print("Warning: Preprocess(W): The image is black and white. Further processing may not be accurate.")
            return None  # Return None for black and white images
            
        # Expand the dimensions and normalize the pixel values
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize the pixel values
        return img_array 
        
    except Exception as e:
        raise ValueError(f"Error during image preprocessing: {str(e)}")

def predict_image_water(img, model_water):
    try:
        preprocessed_image = preprocess_image_water(img)

        # Check if preprocessed_image is empty (black and white image)
        if preprocessed_image is None:
            return {"warning": warning_msg}  # Return a dictionary

        # Continue with prediction for non-black and white images
        prediction = model_water.predict(preprocessed_image)
        class_index = np.argmax(prediction)
        predicted_class = class_labels_of_water[class_index]
        confidence = prediction[0][class_index]
        return predicted_class, confidence
        
    except Exception as e:
        raise ValueError(f"Error during image prediction: {str(e)}")

# ========================== API Routes ===========================

@app.route('/')
def hello():
    return 'Hello, Flask!'

@app.route('/classify', methods=['POST'])
def classify():
    try:
        # Check if the 'image' key is present in request.files
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400

        # Get the file from request.files
        file = request.files['image']

        # Open the image using PIL
        try:
            img = Image.open(io.BytesIO(file.read()))
        except Exception as e:
            raise ValueError(f"Error opening image: {str(e)}")

        # Perform prediction
        prediction_result = predict_image_classification(img, model_classification)

        # Check if there is a warning
        if "warning" in prediction_result:
            return jsonify({"warning": prediction_result["warning"]}), 200

        # Unpack values from prediction_result
        predicted_class, confidence = prediction_result

        # Return the result
        result = {
            "predicted_class": predicted_class,
            "confidence": float(confidence) * 100
        }
        return jsonify(result)

    except ValueError as ve:
        print("ValueError:", str(ve))
        return jsonify({"error": str(ve)}), 400

    except Exception as e:
        print("An error occurred:", str(e))
        return jsonify({"error": "An error occurred"}), 500

@app.route('/disease_detect', methods=['POST'])
def disease_detect():
    try:
        # Check if the 'image' key is present in request.files
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400

        # Get the file from request.files
        file = request.files['image']

        # Open the image using PIL
        try:
            img = Image.open(io.BytesIO(file.read()))
        except Exception as e:
            raise ValueError(f"Error opening image: {str(e)}")

        # Perform prediction
        prediction_result = predict_image_disease(img, model_disease)

        # Check if there is a warning
        if "warning" in prediction_result:
            return jsonify({"warning": prediction_result["warning"]}), 200

        # Unpack values from prediction_result
        predicted_class, confidence = prediction_result

        # Return the result
        result = {
            "predicted_class": predicted_class,
            "confidence": float(confidence) * 100
        }
        return jsonify(result)

    except ValueError as ve:
        print("ValueError:", str(ve))
        return jsonify({"error": str(ve)}), 400

    except Exception as e:
        print("An error occurred:", str(e))
        return jsonify({"error": "An error occurred"}), 500

@app.route('/water_estimation', methods=['POST'])
def water_estimation():
    try:
        # Check if the 'image' key is present in request.files
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400

        # Get the file from request.files
        file = request.files['image']

        # Open the image using PIL
        try:
            img = Image.open(io.BytesIO(file.read()))
        except Exception as e:
            raise ValueError(f"Error opening image: {str(e)}")

        # Perform prediction
        prediction_result = predict_image_water(img, model_water)

        # Check if there is a warning
        if "warning" in prediction_result:
            return jsonify({"warning": prediction_result["warning"]}), 200

        # Unpack values from prediction_result
        predicted_class, confidence = prediction_result

        # Return the result
        result = {
            "predicted_class": predicted_class,
            "confidence": float(confidence) * 100
        }
        return jsonify(result)

    except ValueError as ve:
        print("ValueError:", str(ve))
        return jsonify({"error": str(ve)}), 400

    except Exception as e:
        print("An error occurred:", str(e))
        return jsonify({"error": "An error occurred"}), 500
