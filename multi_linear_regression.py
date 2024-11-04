import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime

# Read the CSV file
df = pd.read_csv("TSLA_stock_data.csv")

# Convert the Date column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Convert the date to a numerical format (e.g., days)
df['Date_numeric'] = (df['Date'] - df['Date'].min()).dt.days

# Define input and output variables
X = df[['Date_numeric', 'Volume']]  # Independent variables: Date_numeric and Volume
y = df['Open']  # Dependent variable: Open price

# Define and train the Linear Regression model
linear_reg = LinearRegression()
linear_reg.fit(X, y)

# Print the intercept and coefficients of the model
print("Intercept (b0):", linear_reg.intercept_)
print("Coefficients (b1, b2):", linear_reg.coef_)

# Make predictions
y_pred = linear_reg.predict(X)

# Plot actual vs. predicted Open prices
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], y, label="Actual Open Prices", color="blue", alpha=0.5)
plt.plot(df['Date'], y_pred, label="Predicted Open Prices (Multi-Linear Regression)", color="red")
plt.xlabel("Date")
plt.ylabel("Open Price")
plt.legend()
plt.show()

# Make a prediction for a future date and volume
# For example, day 2500 from the start and a volume of 400,000,000
future_date = 2500
future_volume = 400000000
predicted_open = linear_reg.predict([[future_date, future_volume]])
print("Predicted Open Price on day 2500 with volume 400000000:", predicted_open[0])
