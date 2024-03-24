from flask import Flask, render_template, request
import numpy as np
from model import load_model
from backend.preprocess import preprocess_image

app = Flask(__name__, template_folder="templets")

# Load the machine learning model
model = load_model()

@app.route('/')
def index():
    return render_template('login.html')

@app.route('/upload/image', methods=['GET', 'POST'])
def upload_image():     

    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file uploaded', 400

        file = request.files['file']
        if file.filename == '':
            return 'No file selected', 400

        file_path = './images/' + file.filename
        file.save(file_path)
        
        # Preprocess the uploaded image
        image = preprocess_image(file_path)

        # Make predictions using the loaded model
        prediction = model.predict(image)

        predicted_index = np.argmax(prediction)
        
        # Define the list of disease names
        disease_names = ['Left Bundle Branch Block', 'Normal', 'Premature Atrial Contraction',
                         'Premature Ventricular Contractions', 'Right Bundle Branch Block', 
                         'Ventricular Fibrillation']
        
        # Get the predicted disease name
        predicted_disease = disease_names[predicted_index]

        # Return the predicted disease name
        return 'Image uploaded and processed successfully. Predicted Disease: {}'.format(predicted_disease)
    
    # Render the upload page for GET requests
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True, port=80)
