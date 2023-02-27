# Random Forest Regressor

This repository contains a Python script that uses machine learning to predict the stock prices of a given company. The script uses the Random Forest Regressor algorithm to predict the stock prices of the next month based on the historical data of the company.

## Dependencies

This script requires the following dependencies:

- numpy
- pandas
- yfinance
- matplotlib
- sklearn

## Usage

To use this script, follow these steps:

1. Install the required dependencies.
2. Open the script in your Python environment.
3. Set the ticker symbol of the company you want to predict in the **`ticker`** variable.
4. Set the start and end dates of the historical data in the **`startdate`** and **`enddate`** variables.
5. Run the script.

The script will download the historical data of the given company, preprocess it, train a Random Forest Regressor model, and predict the stock prices of the next month. The script will also plot a graph showing the actual and predicted stock prices.
