import pandas as pd
import pandas_ta as ta
import yfinance as yf
import logging
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

# logging.basicConfig(format="{asctime} - {levelname} - {message}", style="{",level=logging.INFO)
logging.basicConfig(format="{message}", style="{",level=logging.INFO)

df = pd.DataFrame() # Empty DataFrame

# Run using 4/10 cores of macbook
# df.ta.cores = 4

API_KEY = "PK21WN1YFV30FA4LN29P" 
API_SECRET = "hlG6wHwHzU5QelX06Mkf420G930E4zKXh9BKYYle" 
BASE_URL = "https://paper-api.alpaca.markets/v2"

ALPACA_CREDS = {
    "API_KEY":API_KEY, 
    "API_SECRET": API_SECRET, 
    "PAPER": True
}


SYMBOL = "AAPL"

# Load data
# df = pd.read_csv("data/stock.csv", sep=",")
# OR if you have yfinance installed
# df = df.ta.ticker("aapl")

# Period is used instead of start/end
# Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
# Default: "max"

# History by Interval by interval (including intraday if period < 60 days)
# Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
# Default: "1d"
df = df.ta.ticker(SYMBOL, period="5y", interval="1d") # Gets this past month in hours
# Clean df
df.drop('Stock Splits', axis=1, inplace=True)
df.drop('Dividends', axis=1, inplace=True)


# VWAP requires the DataFrame index to be a DatetimeIndex.
# Replace "datetime" with the appropriate column from your DataFrame
# df.set_index(pd.DatetimeIndex(df["datetime"]), inplace=True)

# Calculate Returns and append to the df DataFrame
# df.ta.log_return(cumulative=True, append=True)
df.ta.percent_return(cumulative=False, append=True)
df["up"] = (df.ta.percent_return(cumulative=False) > 0)


#  Next close
lastCloseDf = df.tail(1)
lastClose = lastCloseDf.iloc[0]['Close']

df['Next Close'] = df['Close'].shift(-1, fill_value=lastClose)


df.ta.inertia(append=True)
df.ta.rsi(append=True)
df.ta.vwap(append=True)
df.ta.cdl_pattern(name=["doji"],append=True)
df['SMA 10'] = df.ta.sma(10)
df['SMA 50'] = df.ta.sma(50)
df['SMA 200'] = df.ta.sma(200)
df['EMA 20'] = df.ta.ema(20)
df['GoldenCross'] = (df['SMA 50'] > df['SMA 200'])
df.ta.obv(append=True)



# df.ta.strategy("Momentum") # Default values for all Momentum indicators
# df.ta.strategy("overlap", length=42) # Override all Overlap 'length' attributes


# logging.info(df.tail(50))
# df.drop('Volume', axis=1, inplace=True)
# df.plot()
# logging.info(help(ta.inertia))



# INFERENCE
# df.drop('High', axis=1, inplace=True)
# df.drop('Low', axis=1, inplace=True)
# df.drop('PCTRET_1', axis=1, inplace=True)
# df.drop('up', axis=1, inplace=True)

# remove object fields from df
# df = df.select_dtypes(exclude=['object'])

# change NAN with mean value
df=df.fillna(df.mean())

X = df.drop('Next Close',axis=1)
y = df['Next Close']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 1000, random_state = 42)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

result=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})



from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# Mean Absolute Error: 1993.2901175839186 # <20%
# Mean Squared Error: 9668487.223350348
# Root Mean Squared Error: 3109.4191134921566

# Calculate the absolute errors
errors = abs(y_pred - y_test)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / y_test)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')
result.sort_index(inplace=True)
print(result.tail(5))
tail = result.tail(155)
tail.plot(figsize=(10, 4))




# predict todays close

today_df = df.tail(1)
today_df = today_df.drop('Next Close',axis=1)

# print(today_df)
pred = regressor.predict(today_df)
print('prediction')
print(pred)

# New Columns with results
# print(df.columns)

# ==================================
#  Trade Strat
# ==================================


from lumibot.brokers import Alpaca
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader
from datetime import datetime

class MLTrader(Strategy): 
    def initialize(self, symbol:str=SYMBOL, cash_at_risk:float=.5): 
        self.symbol = symbol
        self.sleeptime = "24H"
        self.last_trade = None 
        self.cash_at_risk = cash_at_risk
        self.threshold = 1
        self.trail_percent = 2.5

    def position_sizing(self): 
        cash = self.get_cash() 
        last_price = self.get_last_price(self.symbol)
        quantity = round(cash * self.cash_at_risk / last_price,0)
        return cash, last_price, quantity

    def get_predicted_close(self,):
        today = self.get_datetime().strftime("%Y-%m-%d")
        print(today)

        
        today_df = df[today:today]
        today_df = today_df.drop('Next Close',axis=1)

        print(today_df)
        pred = regressor.predict(today_df)
        print(pred)
        return pred
    
    def before_market_closes(self):
        self.on_trading_iteration(self)

    def on_trading_iteration(self):
        cash, last_price, quantity = self.position_sizing() 
        pred_close = self.get_predicted_close()

        diff = pred_close - last_price
        diff_percent = (diff/last_price)*100

        print(f"Diff: {diff}! percent: {diff_percent}")
        print(diff > 0 and diff_percent > self.threshold)
        print(diff < 0 and diff_percent < (self.threshold*-1))

        if cash > last_price: 
            if diff > 0 and diff_percent > self.threshold: 
                if self.last_trade == "sell": 
                    # self.sell_all()
                    print('change short to long')
                order = self.create_order(
                    self.symbol, 
                    quantity, 
                    "buy", 
                    type="bracket", 
                    # take_profit_price=last_price*1.20,
                    trail_percent=self.trail_percent,
                    stop_loss_price=last_price*.97
                )
                self.submit_order(order) 
                self.last_trade = "buy"
            elif diff < 0 and diff_percent < (self.threshold*-1): 
                if self.last_trade == "buy": 
                    # self.sell_all() 
                    print('change long to short')
                order = self.create_order(
                    self.symbol, 
                    quantity, 
                    "sell", 
                    type="bracket", 
                    # take_profit_price=last_price*.8,
                    trail_percent=self.trail_percent, 
                    stop_loss_price=last_price*1.03
                )
                self.submit_order(order) 
                self.last_trade = "sell"

start_date = datetime(2020,8,1)
end_date = datetime(2024,8,19) 
broker = Alpaca(ALPACA_CREDS) 
strategy = MLTrader(name='mlstrat', broker=broker, 
                    parameters={"symbol":SYMBOL, 
                                "cash_at_risk":.8})
# strategy.backtest(
#     YahooDataBacktesting, 
#     start_date, 
#     end_date, 
#     parameters={"symbol":SYMBOL, "cash_at_risk":.8},
#     benchmark_asset="AAPL"
# )

trader = Trader()
trader.add_strategy(strategy)
trader.run_all()