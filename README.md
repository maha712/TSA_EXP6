# Ex.No: 6               HOLT WINTERS METHOD

REG NO: 212222240057

NAME: MAHALAKSHMI K

### Date: 

### AIM:

### ALGORITHM:

1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions

### PROGRAM:
```
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('covid_data.csv', parse_dates=['Date'])
df.set_index('Date', inplace=True)

# Create the model
model = ExponentialSmoothing(df['Cases'], 
                             trend='add', 
                             seasonal='add', 
                             seasonal_periods=7)

# Fit the model
hw_fit = model.fit()

# Make predictions
df['Holt-Winters'] = hw_fit.fittedvalues

# Forecast the next 30 days
forecast = hw_fit.forecast(steps=30)

# Plotting the results
plt.figure(figsize=(14, 7))
plt.plot(df['Cases'], label='Actual Cases', color='blue')
plt.plot(df['Holt-Winters'], label='Holt-Winters Fit', color='orange')
plt.plot(forecast, label='Forecast', color='red')
plt.title('Holt-Winters Method for COVID-19 Cases')
plt.xlabel('Date')
plt.ylabel('Number of Cases')
plt.legend()
plt.show()
```

### OUTPUT:


TEST_PREDICTION

![Screenshot (655)](https://github.com/user-attachments/assets/0893d578-5a9b-41cb-a7ea-05c1a0084935)


FINAL_PREDICTION

![Screenshot (654)](https://github.com/user-attachments/assets/8208a786-3cc7-4507-9523-89c85183be02)

### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
