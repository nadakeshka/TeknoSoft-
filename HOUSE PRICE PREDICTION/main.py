import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


my_data = pd.read_csv("train.csv")
print(my_data.head(5))
print(my_data.info())

print(my_data.isnull().sum())

# numeric and categorical columns
numeric_columns = my_data.select_dtypes(include=np.number).columns
categorical_columns = my_data.select_dtypes(include=object).columns

# Impute missing values
numeric_imputer = SimpleImputer(strategy='mean')
categorical_imputer = SimpleImputer(strategy='most_frequent')

my_data_filled_numeric = pd.DataFrame(numeric_imputer.fit_transform(my_data[numeric_columns]), columns=numeric_columns)
my_data_filled_categorical = pd.DataFrame(categorical_imputer.fit_transform(my_data[categorical_columns]), columns=categorical_columns)

# One-hot encoding
onehotencoder = OneHotEncoder()
encoded_columns = pd.DataFrame(onehotencoder.fit_transform(my_data_filled_categorical).toarray())

# Combine
my_data_encoded = pd.concat([my_data_filled_numeric, encoded_columns], axis=1)
X = my_data_encoded.drop(columns=['MiscVal'], axis=1)
X.columns = X.columns.astype(str)
y = my_data_encoded['MiscVal']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear regression model
model = LinearRegression()


model.fit(X_train, y_train)

# testing data
y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE :", mse)
print("rsore :", r2)

plt.scatter(X_test['BsmtFinSF1'], y_test, color='blue', label='Actual Data')
plt.plot(X_test['BsmtFinSF1'], y_pred, color='red', linewidth=2, label='Regression Line')




plt.xlabel('X Axis Label')
plt.ylabel('Y Axis Label')
plt.title('Regression Line vs Actual Data')
plt.legend()
plt.show()