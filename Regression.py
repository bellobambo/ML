import math
import pandas as pd
import quandl
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Set your Quandl API key
quandl.ApiConfig.api_key = 'qF9Fz96c1QraiskjY7sW'

# Define the dataset you want to fetch from Quandl (e.g., "FRED/NROUST" - US Natural Resources Owned and Consumed by All Sectors, Quarterly)
dataset = "FINRA/FORF_THNCF"

# Fetch the data from Quandl and store it in a Pandas DataFrame
df = quandl.get(dataset)

# Now you can work with the 'df' DataFrame containing the fetched data from Quandl
df = df[['ShortVolume', 'ShortExemptVolume', 'TotalVolume']]
df['HL_PCT'] = (df['TotalVolume'] - df['ShortVolume']) / df["ShortVolume"] * 100.0
df['PCT_change'] = (df['ShortVolume'] - df['TotalVolume']) / df["TotalVolume"] * 100.0

df = df[['ShortVolume', 'HL_PCT', 'PCT_change']]

forecast_col = 'ShortVolume'
df.fillna('-9999', inplace=True)

forecast_out = int(math.ceil(0.01 * len(df)))
print(forecast_out)

df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)

X = np.array(df.drop('label', axis=1))
y = np.array(df['label'])
X = preprocessing.scale(X)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)

print(accuracy)
