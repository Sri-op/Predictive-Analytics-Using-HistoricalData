# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# -----------------------------------
# Create Historical Dataset
# -----------------------------------

data = {
    'Month': [
        'Jan','Feb','Mar','Apr','May','Jun',
        'Jul','Aug','Sep','Oct','Nov','Dec'
    ],

    'Sales': [
        12000,15000,18000,20000,22000,25000,
        27000,30000,32000,35000,37000,40000
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

print("Historical Dataset:")
print(df)

# -----------------------------------
# Data Preprocessing
# -----------------------------------

# Convert Month into Numeric Values
df['Month_Number'] = np.arange(1, 13)

# Features and Target
X = df[['Month_Number']]
y = df['Sales']

# -----------------------------------
# Split Dataset
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------------
# Train Regression Model
# -----------------------------------

model = LinearRegression()

model.fit(X_train, y_train)

# -----------------------------------
# Predict Test Data
# -----------------------------------

y_pred = model.predict(X_test)

# -----------------------------------
# Model Evaluation
# -----------------------------------

mae = mean_absolute_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation")

print("Mean Absolute Error:", mae)

print("R2 Score:", r2)

# -----------------------------------
# Future Prediction
# -----------------------------------

future_months = pd.DataFrame({
    'Month_Number': [13,14,15,16]
})

future_predictions = model.predict(future_months)

print("\nFuture Sales Predictions")

for i, prediction in enumerate(future_predictions):
    print(f"Month {13+i} Predicted Sales: {prediction:.2f}")

# -----------------------------------
# Visualization
# -----------------------------------

plt.figure(figsize=(10,5))

# Actual Sales
plt.plot(
    df['Month_Number'],
    df['Sales'],
    marker='o',
    label='Actual Sales'
)

# Regression Line
plt.plot(
    X,
    model.predict(X),
    color='red',
    label='Predicted Trend'
)

# Future Forecast
plt.plot(
    future_months,
    future_predictions,
    marker='o',
    linestyle='dashed',
    label='Future Forecast'
)

plt.title("Sales Forecast Using Predictive Analytics")

plt.xlabel("Month Number")

plt.ylabel("Sales")

plt.legend()

plt.grid(True)

plt.show()

# -----------------------------------
# Export Prediction Report
# -----------------------------------

forecast_df = pd.DataFrame({
    'Future_Month': [13,14,15,16],
    'Predicted_Sales': future_predictions
})

forecast_df.to_csv("sales_forecast_report.csv", index=False)

print("\nForecast Report Generated Successfully!")
