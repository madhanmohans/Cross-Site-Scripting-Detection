from flask import Flask, render_template, request
from keras.models import load_model
import numpy as np

app = Flask(__name__)
model = load_model('LSTM-v1.h5')  # Load the saved Keras model

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input from the HTML form
    input_data = request.form['input']
    
    # Preprocess the input data
    input_data = np.array([input_data])  # Convert to a numpy array
    input_data = np.expand_dims(input_data, axis=2)  # Add a third dimension
    
    # Make predictions using the Keras model
    prediction = model.predict(input_data)[0][0]
    
    # Round the prediction to the nearest integer
    prediction = int(round(prediction))
    
    # Display the prediction result
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
