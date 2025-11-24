
# Parkinson’s Disease Detection from Spiral and Wave Drawings

Detect Parkinson’s Disease using deep learning models (CNN, DenseNet, VGG16) on spiral and wave drawing images. Includes a Flask web app for easy prediction.

---

## Features
- Train and evaluate models on spiral and wave drawing datasets
- Use CNN, DenseNet, and VGG16 architectures
- Combine predictions for improved accuracy
- Flask web app for image upload and diagnosis

---

## Project Structure

```
├── app.py                # Flask web application
├── CNN_Model.h5          # Trained CNN model
├── spiral_model.h5/.keras # Trained spiral model
├── wave_model.h5/.keras   # Trained wave model
├── dataset/
│   ├── spiral/
│   │   ├── training/
│   │   │   ├── healthy/
│   │   │   └── parkinson/
│   │   └── testing/
│   │       ├── healthy/
│   │       └── parkinson/
│   ├── wave/
│       ├── training/
│       │   ├── healthy/
│       │   └── parkinson/
│       └── testing/
│           ├── healthy/
│           └── parkinson/
├── requirements.txt      # Python dependencies
├── templates/            # HTML templates for Flask
├── static/               # Static files (uploaded images)
├── Final Model.ipynb     # Main training notebook
├── Test Model.ipynb      # Model testing and analysis
```

---

## Setup & Installation

1. Clone the repository:
	 ```
	 git clone <repo-url>
	 cd Parkinson-s-Disease-Detection-from-Spiral-and-Wave-Drawings
	 ```
2. Install dependencies:
	 ```
	 pip install -r requirements.txt
	 ```
3. Place trained models (`CNN_Model.h5`, `spiral_model.keras`, `wave_model.keras`) in the project root.

---

## Running the Web App

1. Start the Flask server:
	 ```
	 python app.py
	 ```
2. Open your browser and go to [http://127.0.0.1:5000/](http://127.0.0.1:5000/)
3. Upload spiral and wave images to get predictions (Healthy or Parkinson)

<img width="1906" height="1027" alt="Image" src="https://github.com/user-attachments/assets/9b3d1c7b-57c7-40a5-b872-8696db10a6fa" />

<img width="1908" height="1026" alt="Image" src="https://github.com/user-attachments/assets/f6ea591b-5037-49b0-98aa-8bbd6de387c2" />
---

## Dataset Structure

Organize your dataset as follows:

```
dataset/
	spiral/
		training/
			healthy/
			parkinson/
		testing/
			healthy/
			parkinson/
	wave/
		training/
			healthy/
			parkinson/
		testing/
			healthy/
			parkinson/
```

---

## Model Training & Testing

- Use `Final Model.ipynb` and `Test Model.ipynb` for training and evaluating models.
- Models use Keras, TensorFlow, and scikit-learn for deep learning and metrics.
- Combined prediction accuracy is calculated by averaging model outputs and comparing to true labels.

---

## Requirements

See `requirements.txt` for all dependencies:
- Flask
- TensorFlow
- numpy
- Pillow
- joblib
- matplotlib

---

## License

This project is for academic and research purposes.
