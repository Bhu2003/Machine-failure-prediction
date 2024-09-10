from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split
import os

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Ensure the uploads directory exists
os.makedirs('uploads', exist_ok=True)

# Function to preprocess and predict
def preprocess_and_predict(filepath):
    try:
        # Load the dataset
        train = pd.read_csv(filepath, index_col=0)
        logging.debug("CSV file loaded successfully.")
        
        # Renaming columns
        train.columns = ['Date', 'Temperature', 'Humidity', 'Operator', 'Measure1', 'Measure2',
                         'Measure3', 'Measure4', 'Measure5', 'Measure6', 'Measure7', 'Measure8', 
                         'Measure9', 'Measure10', 'Measure11', 'Measure12', 'Measure13', 
                         'Measure14', 'Measure15', 'Hours_since_prev', 'Failure', 'Year', 
                         'Month', 'Day', 'Week', 'Hour', 'Minute', 'Second']
        
        # Convert 'Failure' to categorical and then to codes
        train['Failure'] = train['Failure'].astype('category')
        train['Failure'] = train['Failure'].cat.codes
        
        # Cap and floor Humidity and Temperature values using .loc to avoid SettingWithCopyWarning
        train.loc[train['Humidity'] < 70, 'Humidity'] = 70
        train.loc[train['Humidity'] > 95, 'Humidity'] = 95
        train.loc[train['Temperature'] < 60, 'Temperature'] = 60
        train.loc[train['Temperature'] > 71.875, 'Temperature'] = 71.875
        
        # Feature selection based on correlation observation
        X = train[['Temperature', 'Humidity', 'Hours_since_prev', 'Year', 'Month', 'Day', 'Week']]
        y = train['Failure']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)  # Changed to 0.3 for better model evaluation
        
        # Initialize and train the Logistic Regression model
        model = LogisticRegression()
        model.fit(X_train, y_train)
        
        # Make predictions
        predicted_classes = model.predict(X_test)
        
        # Evaluate the model
        mae = mean_absolute_error(y_test, predicted_classes)
        mse = mean_squared_error(y_test, predicted_classes)
        rmse = np.sqrt(mse)
        
        # Add predicted failure to the dataset
        train['Predicted_Failure'] = model.predict(X)
        
        # Calculate days until failure
        train['Date'] = pd.to_datetime(train['Date'])
        train['Days_Until_Failure'] = (train['Date'].max() - train['Date']).dt.days
        failure_estimate = train[train['Predicted_Failure'] == 1]['Days_Until_Failure'].mean()
        
        output_filepath = 'static/train_with_predictions.csv'
        train.to_csv(output_filepath, index=False)
        
        logging.debug("Prediction and evaluation completed successfully.")
        return mae, mse, rmse, failure_estimate, output_filepath
    except Exception as e:
        logging.error(f"Error in preprocess_and_predict: {e}")
        raise

# Home route to upload file
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle file upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        logging.error("No file part in the request.")
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        logging.error("No selected file.")
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        filepath = os.path.join('uploads', file.filename)
        file.save(filepath)
        logging.debug("File uploaded successfully.")
        
        try:
            mae, mse, rmse, failure_estimate, output_filepath = preprocess_and_predict(filepath)
            logging.debug("Preprocessing and prediction completed.")
            return render_template('result.html', 
                                   mae=mae, 
                                   mse=mse, 
                                   rmse=rmse, 
                                   failure_estimate=failure_estimate,
                                   output_filepath=output_filepath)
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
