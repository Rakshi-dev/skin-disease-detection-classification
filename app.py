import os
import numpy as np
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load your trained model
model = load_model('model.h5')
print('Model loaded. Check http://127.0.0.1:5000/')

# Define your labels
labels = {0: 'BA- cellulitis', 1: 'BA-impetigo', 2: 'FU-athlete-foot', 3: 'FU-nail-fungus', 4:'FU-ringworm', 5:'PA-cutaneous-larva-migrans', 6:'VI-chickenpox', 7:'VI-shingles'}

# Function to preprocess the image and get predictions
def getResult(image_path):
    img = load_img(image_path, target_size=(224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x.astype('float32') / 255.0  # Normalize pixel values
    predictions = model.predict(x)[0]
    return predictions

# Define your Flask routes
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        # Get the file from the POST request
        f = request.files['file']

        # Save the file to a location
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Get predictions
        predictions = getResult(file_path)
        pred_class = np.argmax(predictions)
        predicted_label = labels[pred_class]

        # Check if the predicted label is 'B12 deficiency'
        if predicted_label == 'B12 deficiency':
            suggestions_link = '<a href="https://www.webmd.com/diet/b12-rich-foods" target="_blank">Suggestions to improve B12 levels</a>'
        else:
            suggestions_link = ''

        return render_template('result.html', prediction=predicted_label, link=suggestions_link)

    return 'Invalid request method'

if __name__ == '__main__':
    app.run(debug=True)
