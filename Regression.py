
import math
import pandas as pd
import quandl

# Set your Quandl API key
quandl.ApiConfig.api_key = 'qF9Fz96c1QraiskjY7sW'

# Define the dataset you want to fetch from Quandl (e.g., "FRED/NROUST" - US Natural Resources Owned and Consumed by All Sectors, Quarterly)
dataset = "FINRA/FORF_THNCF"

# Fetch the data from Quandl and store it in a Pandas DataFrame
df = quandl.get(dataset)

# Now you can work with the 'df' DataFrame containing the fetched data from Quandl
df = df[['ShortVolume' , 'ShortExemptVolume' , 'TotalVolume' ,]]
df['HL_PCT'] = (df['TotalVolume'] - df['ShortVolume']) / df["ShortVolume"] * 100.0
df['PCT_change'] = (df['ShortVolume'] - df['TotalVolume']) / df["TotalVolume"] * 100.0


df = df[['ShortVolume' , 'HL_PCT' , 'PCT_change']]

forecast_col = 'ShortVolume'
df.fillna('-9999, inplace=True')

forecast_out = int(math.ceil(0.01*len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)
print(df.tail)