from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load dataset
df = pd.read_csv("Chennai houseing sale.csv")

# Fill missing values
df = df.fillna(df.mean(numeric_only=True))

# Select useful columns (including location)
features = ['INT_SQFT', 'N_BEDROOM', 'N_BATHROOM', 'DIST_MAINROAD', 'AREA']

# Convert categorical (AREA = location)
df = pd.get_dummies(df[features + ['SALES_PRICE']], drop_first=True)

# Separate X and y
X = df.drop('SALES_PRICE', axis=1)
y = df['SALES_PRICE']

# Train model
model = LinearRegression()
model.fit(X, y)

# Get all location names
locations = [col.replace("AREA_", "") for col in X.columns if "AREA_" in col]

@app.route('/')
def home():
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        area = float(request.form['area'])
        bedroom = int(request.form['bedroom'])
        bathroom = int(request.form['bathroom'])
        distance = float(request.form['distance'])
        location = request.form['location']

        # Create input with all zeros
        input_data = [0] * len(X.columns)

        columns = list(X.columns)

        # Set numeric values
        input_data[columns.index('INT_SQFT')] = area
        input_data[columns.index('N_BEDROOM')] = bedroom
        input_data[columns.index('N_BATHROOM')] = bathroom
        input_data[columns.index('DIST_MAINROAD')] = distance

        # Set location (one-hot)
        loc_col = "AREA_" + location
        if loc_col in columns:
            input_data[columns.index(loc_col)] = 1

        input_df = pd.DataFrame([input_data], columns=columns)

        prediction = model.predict(input_df)[0]

        return render_template('index.html',
                               prediction_text=f"Estimated Price: ₹ {int(prediction)}",
                               locations=locations)

    except Exception as e:
        return render_template('index.html',
                               prediction_text="Error: Check inputs",
                               locations=locations)

if __name__ == "__main__":
    app.run(debug=True)