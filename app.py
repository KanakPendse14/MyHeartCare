import pickle
from altair import Scale
from flask import Flask, render_template, request
import joblib
from matplotlib import scale
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model

# from tensorflow.keras.models import load_model as load_h5_model
from model import load_model
from backend.preprocess import preprocess_image
import pickle

app = Flask(__name__, template_folder="templets")


@app.route('/')
def index():
    return render_template('home.html')


# model 2(CNN image processing)
# model = load_model('./CNN/ECG.h5')
new_model = tf.keras.models.load_model('./CNN/ECG.h5')

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
        
        image = preprocess_image(file_path)

        prediction = new_model.predict(image)

        predicted_index = np.argmax(prediction)
        
        # list of disease names
        disease_names = ['Left Bundle Branch Block', 'Normal', 'Premature Atrial Contraction',
                         'Premature Ventricular Contractions', 'Right Bundle Branch Block', 
                         'Ventricular Fibrillation']
        
        predicted_disease = disease_names[predicted_index]
        

        return render_template('upload.html', predicted_disease=predicted_disease)
        # return 'Image uploaded and processed successfully. Predicted Disease: {}'.format(predicted_disease)
    
    return render_template('upload.html')

#next model 2(Diabetes)
with open('CNN/model.pkl', 'rb') as f:
    model = pickle.load(f)

def predict_diabetes(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    prediction = model.predict(input_data)
    return prediction[0]

@app.route('/predict/diabetes', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        pregnancies = float(request.form['pregnancies'])
        glucose = float(request.form['glucose'])
        blood_pressure = float(request.form['blood_pressure'])
        skin_thickness = float(request.form['skin_thickness'])
        insulin = float(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = int(request.form['dpf'])
        age = float(request.form['age'])

        prediction = predict_diabetes(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age)
        
        if prediction == 1:
            result = 'Our AI analysis suggests that the user may already have diabetes or could be at high risk of developing it in the near future.'
        else:
            result = 'Our AI analysis indicates that the user is currently healthy and should continue their lifestyle to maintain good health and avoid the risk of developing diabetes.'
        
        return render_template('diabetespredictpage.html', result=result)
    

    return render_template('diabetespredictpage.html') 


#model 3(Heart Health)

heart_model = joblib.load('CNN/HeartDiseaseprediction.sav')

def predict_heart_disease(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    prediction = model.predict(input_data)
    return prediction[0]

@app.route('/predict/hearthealth', methods=['GET' ,'POST'])
def predict_hearthhealth():
    if request.method == 'POST':
        try:
          
            age = int(request.form['age'])
            sex = int(request.form['sex'])
            cp = int(request.form['cp'])
            trestbps = int(request.form['trestbps'])
            chol = int(request.form['chol'])
            fbs = int(request.form['fbs'])
            restecg = int(request.form['restecg'])
            thalach = int(request.form['thalach'])
            exang = int(request.form['exang'])
            oldpeak = float(request.form['oldpeak'])
            slope = int(request.form['slope'])
            ca = int(request.form['ca'])
            thal = int(request.form['thal'])

            input_data = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]).reshape(1, -1)

            prediction = heart_model.predict(input_data)

            if prediction[0] == 1:
                result = 'AI has predicted that the patient has a heart condition and needs further medical examination to assess whether it is at an early stage or already developed condition.'
            else:
                result = 'AI has determined that the patient does not have a heart condition. However, it is important for them to maintain a healthy lifestyle to prevent any potential future risks.'

            return render_template('hearthealthpage.html', result=result)
        except KeyError:

            error_message = "Form data is incomplete. Please fill out all required fields."
            return render_template('hearthealthpage.html', error_message=error_message)
    
    return render_template('hearthealthpage.html')



@app.route('/home/page')
def home_page():
    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True, port=80)
