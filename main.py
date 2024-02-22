# Importing the dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import dataset
house_prediction_data = pd.read_csv("C:/Users/ajith/Downloads/Data (1).csv")
print(house_prediction_data)

# Print first five column
print(house_prediction_data.head())

# Add target column(price) to the dataset
print(house_prediction_data.columns)
house_prediction_data.rename(columns={'MEDV': 'Prices'}, inplace=True)
print(house_prediction_data.head())

# Checking the number of rows and columns

print(house_prediction_data.shape)

# Checking for missing values

missing_values = house_prediction_data.isnull().sum()
print(missing_values)

# Statistical measures of the dataset
description = house_prediction_data.describe()
print(description)

# Understanding the correlation between various features in the dataset

correlation = house_prediction_data.corr()
print(correlation)

# Constructing a heatmap to understand the correlation
plt.figure(figsize=(10, 10))

sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size': 8}, cmap='Blues')
plt.title("Correlation ")

plt.show()

# Split the data and target
x = house_prediction_data.drop('Prices', axis=1)
y = house_prediction_data['Prices']

print(x)
print(y)

# Split the data into training dat and test data
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(x.shape, x_train.shape, x_test.shape)

# Model training
from xgboost import XGBRegressor

model =XGBRegressor()

# Training the model
model.fit(x_train,y_train)

# Evaluation
# Predict on the training set
train_pred = model.predict(x_train)
print(train_pred)

# R_square error
from sklearn import metrics

r2 = metrics.r2_score(y_train, train_pred)
print("R-squared score on training set:", r2)

# Mean absolute error
mse = metrics.mean_squared_error(y_train, train_pred)
print("Mean Squared Error:", mse)

# Visualizing the actual prices and predicted prices
plt.scatter(y_train,train_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Price vs Predicted Price")
plt.show()

# Prediction on the test data

test_pred = model.predict(x_test)
print(test_pred)

# R_square error
r2 = metrics.r2_score(y_test, test_pred)
print("R-squared score on test set:", r2)

# Mean absolute error
mse = metrics.mean_squared_error(y_test, test_pred)
print("Mean Squared Error:", mse)
