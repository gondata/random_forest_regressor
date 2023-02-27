# First of all we have to import the libraries that we are going to use

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import sklearn.preprocessing as preprocessing # The "preprocessing" submodule includes various data preprocessing techniques for feature transformation, normalization, encoding, and scaling.
from pandas_datareader import data as pdr
from sklearn.model_selection import train_test_split # This function is commonly used to split a dataset into training and testing sets.
from sklearn.ensemble import RandomForestRegressor

yf.pdr_override()

# Then we have to download the data

ticker = ['TSLA']
startdate = '2020-01-01'
enddate = '2023-02-18'

data = pdr.get_data_yahoo(ticker, start=startdate, end=enddate)

# Preparing the data

df = data.reset_index() # New numeric index
stock = data['Adj Close']

# Preparing the data

df['HL_Perc'] = (df['High']-df['Low']) / df['Low'] * 100 # % Variation between daily High and Low
df['CO_Perc'] = (df['Close']-df['Open']) / df['Open'] * 100 # % Variation between daily Open and Close

dates = np.array(df['Date']) # We create an array with the dates
dates_c = dates[-30:] # Last 30 days
dates = dates[:-30] # Everyday - last 30 days

# We indicate the model which data is independant and which one is dependant

# Independent variables

df = df[['HL_Perc', 'CO_Perc', 'Volume', 'Adj Close']] # We select the columns that we are going to use

# Dependent variable

df['PriceNextMonth'] = df['Adj Close'].shift(-30) # We create a new column for the stock price in a month.

# Independant variable

X = np.array(df.drop(['PriceNextMonth'], 1)) # We create an array with the independent variables
X = preprocessing.scale(X) # We normalize the data. N(0, 1)
X = X[:-30] # We drop the last 30 rows.
X_c = X[-30:] # We assign the last 30 rows to another variable.

# Cleaning the data

df.dropna(inplace=True) # We drop all the rows with missing values.

# Dependant variable

y = np.array(df['PriceNextMonth']) # We create an array with the dependent variables.

# Then we have to separate the data between the data that we are going to use to train the model and the data that we are going to use to test it

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # We use the train_test_split function to split our train and test data. 70/30

# Model: Training data

rfr = RandomForestRegressor() # We build a random forest regressor model
rfr.fit(X_train, y_train) # We train the model

# Model: Testing data

conf = rfr.score(X_test, y_test) # We calculate the precision
rfr.fit = (X, y)

predictions = rfr.predict(X_c)

# Creating new df

actual = pd.DataFrame(dates, columns = ['Date'])
actual['Adj Close'] = df['Adj Close'] #Real data of Adj Close
actual['PriceNextMonth'] = np.nan
actual.set_index('Date', inplace = True)

forecast = pd.DataFrame(dates_c, columns = ['Date'])
forecast['PriceNextMonth'] = predictions # Forecast predictions
forecast['Adj Close'] = np.nan
forecast.set_index('Date', inplace = True)

var = [actual, forecast]
result = pd.concat(var)

# Graphs

# Real 

stock.plot(figsize=(12, 6))
plt.ylabel('Price')
plt.xlabel('Date')

# Forecast

result.plot(figsize=(12, 6))
plt.title('Random Forest Regressor')
plt.ylabel('Price')
plt.xlabel('Date')
plt.show()