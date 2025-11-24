from flask import Flask, request, render_template, jsonify
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt
import io
import base64
import os

app = Flask(__name__)

# Label mappings
Name = ['healthy', 'parkinson']
normal_mapping = dict(zip(Name, range(len(Name))))
reverse_mapping = dict(zip(range(len(Name)), Name))

def mapper(value):
    return reverse_mapping[value]

# Single image prediction function
def predict_image(model, image, target_size=(224,224), is_spiral=True):
    try:
        print(f"Processing image with target_size={target_size}, is_spiral={is_spiral}")
        img = image.resize(target_size)
        if is_spiral:
            img = img.convert('L')  # Convert to grayscale for spiral
        img_array = img_to_array(img)
        print(f"Image array shape: {img_array.shape}")
        if is_spiral:
            img_array = np.repeat(img_array, 3, axis=-1)  # Convert grayscale to RGB
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        pred = model.predict(img_array, verbose=0)
        print(f"Prediction probabilities: {pred}")
        label = mapper(np.argmax(pred))
        return label, pred[0], img, None
    except Exception as e:
        print(f"Error in predict_image: {str(e)}")
        return None, None, None, str(e)

# Combined prediction function
def predict_combined(spiral_model, wave_model, spiral_image, wave_image):
    try:
        print("Starting combined prediction")
        spiral_label, spiral_probs, spiral_img, spiral_error = predict_image(spiral_model, spiral_image, target_size=(224,224), is_spiral=True)
        wave_label, wave_probs, wave_img, wave_error = predict_image(wave_model, wave_image, target_size=(100,100), is_spiral=False)
        if spiral_label is None or wave_label is None:
            error = spiral_error or wave_error or "Prediction failed for one or both images"
            print(f"Combined prediction error: {error}")
            return None, error
        avg_probs = (spiral_probs + wave_probs) / 2
        final_label = np.argmax(avg_probs)
        final_diagnosis = mapper(final_label)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(spiral_img, cmap='gray')
        ax1.set_title(f"Spiral: {spiral_label}\nHealthy: {spiral_probs[0]:.3f}, Parkinson: {spiral_probs[1]:.3f}")
        ax1.axis('off')
        ax2.imshow(wave_img)
        ax2.set_title(f"Wave: {wave_label}\nHealthy: {wave_probs[0]:.3f}, Parkinson: {wave_probs[1]:.3f}")
        ax2.axis('off')
        plt.suptitle(f"Final Diagnosis: {final_diagnosis} (Healthy: {avg_probs[0]:.3f}, Parkinson: {avg_probs[1]:.3f})")
        
        # Convert plot to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        return {
            'final_diagnosis': final_diagnosis,
            'final_probabilities': {'healthy': float(avg_probs[0]), 'parkinson': float(avg_probs[1])},
            'spiral_prediction': spiral_label,
            'spiral_probabilities': {'healthy': float(spiral_probs[0]), 'parkinson': float(spiral_probs[1])},
            'wave_prediction': wave_label,
            'wave_probabilities': {'healthy': float(wave_probs[0]), 'parkinson': float(wave_probs[1])},
            'plot': plot_base64
        }, None
    except Exception as e:
        print(f"Error in predict_combined: {str(e)}")
        return None, str(e)

# Load models
try:
    print("Loading models...")
    spiral_model = load_model('spiral_model.keras', compile=False)
    wave_model = load_model('wave_model.keras', compile=False)
    print("Models loaded successfully")
except Exception as e:
    print(f"Error loading models: {str(e)}")
    spiral_model = wave_model = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if spiral_model is None or wave_model is None:
        print("Models not loaded")
        return jsonify({'error': 'Models not loaded. Ensure spiral_model.keras and wave_model.keras are in the directory.'}), 500
    
    if 'spiral' not in request.files or 'wave' not in request.files:
        print("Missing files in request")
        return jsonify({'error': 'Please upload both spiral and wave images.'}), 400
    
    spiral_file = request.files['spiral']
    wave_file = request.files['wave']
    
    if spiral_file.filename == '' or wave_file.filename == '':
        print("Empty file names")
        return jsonify({'error': 'Please select valid image files.'}), 400
    
    try:
        print(f"Processing files: {spiral_file.filename}, {wave_file.filename}")
        spiral_image = Image.open(spiral_file)
        wave_image = Image.open(wave_file)
        
        # Get base64 for uploaded images
        spiral_buf = io.BytesIO()
        spiral_image.save(spiral_buf, format='PNG')
        spiral_base64 = base64.b64encode(spiral_buf.getvalue()).decode('utf-8')
        
        wave_buf = io.BytesIO()
        wave_image.save(wave_buf, format='PNG')
        wave_base64 = base64.b64encode(wave_buf.getvalue()).decode('utf-8')
        
        # Make prediction
        result, error = predict_combined(spiral_model, wave_model, spiral_image, wave_image)
        if error:
            print(f"Prediction error: {error}")
            return jsonify({'error': error}), 500
            
        # Return results with uploaded image base64
        result['spiral_image'] = spiral_base64
        result['wave_image'] = wave_base64
        print("Prediction successful")
        return jsonify(result)
    except Exception as e:
        print(f"Error in predict endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)