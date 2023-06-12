from flask import Flask, request, render_template
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the dataset
dataset = pd.read_csv('C:\\Users\\abdul\\Downloads\\New folder\\data.csv')
df = pd.DataFrame(dataset)

# Split the data into features and target variable
dataset.columns[dataset.isna().any()]
dataset.hours = dataset.hours.fillna(dataset.hours.mean())
X = df.drop('marks', axis=1)
y = df['marks']

# Train the model
model = LinearRegression()
model.fit(X, y)

app = Flask(__name__, template_folder='templates')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    hours = float(request.form['hours'])
    age = float(request.form['age'])
    internet = float(request.form['internet'])
    new_cust = [[hours, age, internet]]
    result_value = model.predict(new_cust)
    prediction_text = result_value[0]
    return render_template('index.html', prediction=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
