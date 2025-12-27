# Skin Disease Detection and Classification

This project is a Machine Learning–based web application for detecting and classifying skin diseases from images.  
It helps identify common skin conditions using a trained deep learning model.

---

## Features
- Upload skin images
- Predict skin disease
- Simple and user-friendly web interface
- Built using Flask and Deep Learning

---

## Technologies Used
- Python
- Flask
- TensorFlow / Keras
- NumPy
- OpenCV
- HTML, CSS

---

```text
Skin disease detection and classification/
├── app.py
├── templates/
├── static/
├── uploads/
├── dataset/
├── model.h5
├── best_weights.h5
└── README.md

## How to Run the Project
1. Clone the repository
git clone https://github.com/Rakshi-dev/skin-disease-detection-classification.git
cd skin-disease-detection-classification

2. Create virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate

3. Install required packages
pip install -r requirements.txt
If requirements.txt is not available:
pip install tensorflow keras numpy pandas opencv-python matplotlib scikit-learn flask

4. Run the application
python app.py

5. Open in browser
http://127.0.0.1:5000
