import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from datetime import datetime

# Read the CSV file
df = pd.read_csv("TSLA_stock_data.csv")

# Convert the Date column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Convert the date to a numerical format (e.g., days)
df['Date_numeric'] = (df['Date'] - df['Date'].min()).dt.days

# Define the input and output variables
X = df[['Date_numeric', 'Volume']]  # Independent variables: Date_numeric and Volume
y = df['Open']  # Dependent variable: Open price

# Set the polynomial degree (e.g., 2 for quadratic, 3 for cubic)
degree = 2
poly = PolynomialFeatures(degree=degree)

# Transform the input data to include polynomial terms
X_poly = poly.fit_transform(X)

# Define and train the Polynomial Linear Regression model
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)

# Print the intercept and coefficients of the model
print("Intercept (b0):", poly_reg.intercept_)
print("Coefficients (for polynomial features):", poly_reg.coef_)

# Make predictions
y_pred = poly_reg.predict(X_poly)

# Plot actual vs. predicted Open prices
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], y, label="Actual Open Prices", color="blue", alpha=0.5)
plt.plot(df['Date'], y_pred, label=f"Predicted Open Prices (Polynomial Degree {degree})", color="red")
plt.xlabel("Date")
plt.ylabel("Open Price")
plt.legend()
plt.show()

# Make a prediction for a future date and volume using polynomial terms
# For example, day 2500 from the start and a volume of 400,000,000
future_date = 2500
future_volume = 400000000
future_data_poly = poly.transform([[future_date, future_volume]])
predicted_open = poly_reg.predict(future_data_poly)
print(f"Predicted Open Price on day 2500 with volume 400000000 (Polynomial Degree {degree}):", predicted_open[0])
