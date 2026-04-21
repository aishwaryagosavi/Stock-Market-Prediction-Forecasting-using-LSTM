# Stock Market Prediction Using Stacked LSTM

## Overview
Deep learning model to predict and forecast stock prices using 
a 3-layer Stacked LSTM neural network. Trained on 2,035 days of 
real stock data from Tata Global Beverages (NSE India).

## Results
| Metric | Value |
|---|---|
| Train RMSE | 3.19 |
| Test RMSE | 9.21 |
| Average Stock Price | INR 149.45 |
| Error as % of average price | 6.16% |

The model predicts stock prices within 6.16% of the actual 
average price — a strong result for financial time series prediction.

## 30-Day Price Forecast
Model generates rolling 30-day forecasts based on the last 
100 days of price data. Forecast direction varies between 
runs due to random weight initialisation — reflecting the 
genuine uncertainty in stock price prediction.

Note: This model is for educational purposes. Real stock 
prediction requires additional features like trading volume, 
market sentiment, and macroeconomic indicators.

## How It Works
1. Loaded 2,035 days of historical stock price data
2. Scaled closing prices between 0 and 1 using MinMaxScaler
3. Created sequences of 100 days to predict the next day's price
4. Split data 70/30 into train and test sets
5. Trained a 3-layer Stacked LSTM for 60 epochs
6. Evaluated using RMSE on actual price values
7. Forecasted next 30 days using rolling predictions

## Model Architecture
3 stacked LSTM layers — each layer learns increasingly 
complex temporal patterns in the stock price sequence.

| Layer | Type | Units |
|---|---|---|
| 1 | LSTM | 50 |
| 2 | LSTM | 50 |
| 3 | LSTM | 50 |
| 4 | Dense | 1 |

Total trainable parameters: 50,851

## Why Stacked LSTM?
Stock prices are time series data — today's price depends on 
previous days. LSTM networks are designed to learn from sequential 
data and remember long-term patterns. Stacking 3 LSTM layers allows 
the model to learn simple patterns in layer 1 and progressively 
more complex patterns in layers 2 and 3.

## Why MinMaxScaler?
Neural networks train better on data scaled between 0 and 1.
Raw stock prices would cause unstable gradients during training.
After prediction, values are inverse transformed back to actual 
INR prices.

## Tools and Libraries
- Python
- TensorFlow / Keras (Stacked LSTM)
- NumPy, pandas
- scikit-learn (MinMaxScaler, RMSE)
- matplotlib

## Key Skills Demonstrated
- Deep learning with TensorFlow and Keras
- Time series forecasting with LSTM
- Feature scaling and inverse transformation
- Rolling window prediction for future forecasting
- Financial data analysis
- Understanding and communicating model limitations

## Dataset
2,035 days of Tata Global Beverages stock data from NSE
(National Stock Exchange of India)
