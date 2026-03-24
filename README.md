# Ex.No: 08     MOVINTG AVERAGE MODEL AND EXPONENTIAL SMOOTHING
### Reg No: 212223240081


### AIM:
To implement Moving Average Model and Exponential smoothing Using Python.
### ALGORITHM:
1. Import necessary libraries
2. Read the electricity time series data from a CSV file,Display the shape and the first 20 rows of
the dataset
3. Set the figure size for plots
4. Suppress warnings
5. Plot the first 50 values of the 'Value' column
6. Perform rolling average transformation with a window size of 5
7. Display the first 10 values of the rolling mean
8. Perform rolling average transformation with a window size of 10
9. Create a new figure for plotting,Plot the original data and fitted value
10. Show the plot
11. Also perform exponential smoothing and plot the graph
### PROGRAM:
```
# === Import Libraries ===
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import zipfile
import warnings
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

# === 1. Extract ZIP and Read CSV ===
zip_path = "/content/housing_price_dataset.csv (1).zip"


with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall("extracted_housing_data")
    file_names = zip_ref.namelist()

print("Extracted files:", file_names)

# Load the first CSV automatically
csv_path = "extracted_housing_data/" + file_names[0]
data = pd.read_csv(csv_path)

# === 2. Prepare Dataset ===
print("Columns in dataset:", data.columns.tolist())

# Ensure required columns exist
if 'YearBuilt' not in data.columns:
    raise ValueError("Column 'YearBuilt' not found in dataset!")
if 'Price' not in data.columns:
    raise ValueError("Column 'Price' not found in dataset!")

# Convert YearBuilt into datetime
data['YearBuilt'] = pd.to_datetime(data['YearBuilt'].astype(int), format='%Y')

# Set YearBuilt as index and sort
data = data.set_index('YearBuilt').sort_index()

# Keep only YearBuilt and Price columns
target_data = data[['Price']]

# === 3. Display Info ===
print("Shape of dataset:", target_data.shape)
print("\nFirst 10 rows:")
print(target_data.head(10))

# === 4. Plot Original Data ===
plt.figure(figsize=(12, 6))
plt.plot(target_data, label='Original Price Data', color='blue')
plt.title('Original Housing Price Time Series (by YearBuilt)')
plt.xlabel('YearBuilt')
plt.ylabel('Price')
plt.legend()
plt.grid()
plt.show()

# === 5. Moving Averages ===
rolling_mean_5 = target_data['Price'].rolling(window=5).mean()
rolling_mean_10 = target_data['Price'].rolling(window=10).mean()

print("\nFirst 10 values of rolling mean (window=5):")
print(rolling_mean_5.head(10))

print("\nFirst 20 values of rolling mean (window=10):")
print(rolling_mean_10.head(20))

# Plot moving averages
plt.figure(figsize=(12, 6))
plt.plot(target_data['Price'], label='Original', color='blue')
plt.plot(rolling_mean_5, label='MA (5)', color='orange')
plt.plot(rolling_mean_10, label='MA (10)', color='green')
plt.title('Moving Average (YearBuilt vs Price)')
plt.xlabel('YearBuilt')
plt.ylabel('Price')
plt.legend()
plt.grid()
plt.show()

# === 6. Resample to Yearly Frequency ===
data_yearly = target_data.resample('Y').mean()

# === 7. Scale Data ===
scaler = MinMaxScaler()
scaled_data = pd.Series(
    scaler.fit_transform(data_yearly.values.reshape(-1, 1)).flatten(),
    index=data_yearly.index
)
scaled_data = scaled_data + 1e-3  # for multiplicative stability

# === 8. Train-Test Split ===
split_index = int(len(scaled_data) * 0.8)
train_data = scaled_data[:split_index]
test_data = scaled_data[split_index:]

# === 9. Holt-Winters Model ===
model = ExponentialSmoothing(
    train_data, trend='add', seasonal='mul', seasonal_periods=5
).fit()

predictions = model.forecast(steps=len(test_data))

# Plot Results
ax = train_data.plot(label='Train', figsize=(12, 6))
test_data.plot(ax=ax, label='Test')
predictions.plot(ax=ax, label='Predictions', color='red')
plt.title('Holt-Winters Forecast (Housing Prices)')
plt.xlabel('YearBuilt')
plt.ylabel('Scaled Price')
plt.legend()
plt.grid()
plt.show()

# === 10. RMSE Evaluation ===
rmse = np.sqrt(mean_squared_error(test_data, predictions))
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

# === 11. Forecast Future Prices ===
model_full = ExponentialSmoothing(
    scaled_data, trend='add', seasonal='mul', seasonal_periods=5
).fit()

future_steps = int(len(data_yearly) / 4)
future_forecast = model_full.forecast(steps=future_steps)

# Plot future forecast
ax = scaled_data.plot(label='Historical', figsize=(12, 6))
future_forecast.plot(ax=ax, label='Forecast', color='red')
plt.title('Forecast of Future Housing Prices (Next Quarter)')
plt.xlabel('YearBuilt')
plt.ylabel('Scaled Price')
plt.legend()
plt.grid()
plt.show()
```
### OUTPUT:
<img width="1360" height="681" alt="500835962-0e1d910f-52e4-437c-b3da-e08d5417fac6" src="https://github.com/user-attachments/assets/bcf2a198-a5a8-4306-994a-4e240af290ef" />
<img width="1343" height="674" alt="500836329-43a40d26-2735-45f9-aa52-c770a3445ddd" src="https://github.com/user-attachments/assets/d9703e7e-a0c9-4a7a-b01e-a1260de2bc2d" />
<img width="1313" height="663" alt="500836543-7481ba0d-9d3f-48b8-98c6-c88ef6db9cd2" src="https://github.com/user-attachments/assets/3f1a0ce2-c86d-4a60-83b1-70f7afab2747" />
<img width="1302" height="676" alt="500836732-6d21aedb-4d50-4ccd-8d54-7851c98a20b9" src="https://github.com/user-attachments/assets/e58cd5a4-c7f6-431c-85e9-da1f59a3dfaa" />


### RESULT:
Thus we have successfully implemented the Moving Average Model and Exponential smoothing using python.
