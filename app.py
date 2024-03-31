from flask import Flask, render_template, request
import numpy as np
from sklearn.base import ClassifierMixin
from model import load_model
from backend.preprocess import preprocess_image

app = Flask(__name__, template_folder="templets")

# Load the machine learning model
model = load_model()

@app.route('/')
def index():
    return render_template('home.html')

# Route to render the diabetes prediction page
@app.route('/diabetes_predict', methods=['GET'])
def diabetes_predict():
    # Extract input data from the form
    data = request.form
    features = [data['pregnancies'], data['glucose'], data['blood_pressure'],
                data['skin_thickness'], data['insulin'], data['bmi'],
                data['diabetes_pedigree_function'], data['age']]
    
    # Reshape the features array for prediction
    input_data = np.array(features).reshape(1, -1)

    # Make predictions using the diabetes model
    # Here, you need to use your trained diabetes prediction model
    # Replace 'prediction' with actual predictions from your model
    prediction = Classifier  # Replace this with your model prediction code

    # Return the prediction result
    return render_template('diabetes_prediction_result.html', prediction=prediction)
    # return render_template('diabetespredictpage.html')


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

@app.route('/home/page')
def home_page():
    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True, port=80)
