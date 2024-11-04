import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
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

# Define and train the Decision Tree Regressor model
tree_reg = DecisionTreeRegressor(random_state=42, max_depth=5)
tree_reg.fit(X, y)

# Make predictions
y_pred = tree_reg.predict(X)

# Plot actual vs. predicted Open prices
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], y, label="Actual Open Prices", color="blue", alpha=0.5)
plt.plot(df['Date'], y_pred, label="Predicted Open Prices (Decision Tree Regression)", color="red")
plt.xlabel("Date")
plt.ylabel("Open Price")
plt.legend()
plt.show()

# Make a prediction for a future date and volume
# For example, day 2500 from the start and a volume of 400,000,000
future_date = 2500
future_volume = 400000000
predicted_open = tree_reg.predict([[future_date, future_volume]])
print("Predicted Open Price on day 2500 with volume 400000000:", predicted_open[0])
