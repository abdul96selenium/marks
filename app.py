import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from flask import Flask, request, jsonify

# Load the dataset
data = pd.read_csv('C:\\Users\\data.csv')
df = pd.DataFrame(data)

# Split the data into features and target variable
dataset.columns[dataset.isna().any()]
dataset.hours = dataset.hours.fillna(dataset.hours.mean())
X = df.drop('marks', axis=1)
y = df['marks']

# Train the model
model = LinearRegression()
model.fit(X, y)

# Create Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the request
    user_input = request.json

    # Prepare the user input for prediction
    user_df = pd.DataFrame(user_input)
    
    # Make predictions using the trained model
    predictions = model.predict(user_df)

    # Return the predictions as a JSON response
    return jsonify({'predictions': predictions.tolist()})

if __name__ == '__main__':
    app.run()
