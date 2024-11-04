import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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

# Define and train the Random Forest Regressor model
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X, y)

# Make predictions
y_pred = rf_reg.predict(X)

# Plot actual vs. predicted Open prices
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], y, label="Actual Open Prices", color="blue", alpha=0.5)
plt.plot(df['Date'], y_pred, label="Predicted Open Prices (Random Forest Regression)", color="red")
plt.xlabel("Date")
plt.ylabel("Open Price")
plt.legend()
plt.title("Actual vs Predicted Open Prices")
plt.show()

# Evaluate the regression model
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# Make a prediction for a future date and volume
# For example, day 2500 from the start and a volume of 400,000,000
future_date = 2500
future_volume = 400000000
predicted_open = rf_reg.predict([[future_date, future_volume]])
print("Predicted Open Price on day 2500 with volume 400000000:", predicted_open[0])
