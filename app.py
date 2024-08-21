import os
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import joblib
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from dotenv import load_dotenv
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

# logging.basicConfig(format="{asctime} - {levelname} - {message}", style="{",level=logging.INFO)
logging.basicConfig(format="{message}", style="{",level=logging.INFO)
logging.info('%s before you %s', 'Look', 'leap!')

# Load Env vars
load_dotenv()

df = pd.DataFrame() # Empty DataFrame

API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
BASE_URL = os.getenv("BASE_URL")

if API_KEY is None:
    API_KEY = os.environ.get('API_KEY')
    API_SECRET = os.environ.get('API_SECRET')
    BASE_URL = os.environ.get('BASE_URL')

ALPACA_CREDS = {
    "API_KEY":API_KEY, 
    "API_SECRET": API_SECRET, 
    "PAPER": True
}

symbols = ['AAPL','^GSPC','BTC-USD', 'ETH-USD', '^GDAXI','GC=F']
SYMBOL = symbols[5]
print('Symbol:',SYMBOL)

# Load data

# Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
# Default: "max"

# History by Interval by interval (including intraday if period < 60 days)
# Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
# Default: "1d"
df = df.ta.ticker(SYMBOL, period="5y", interval="1d")




def build_dataFrame(fresh_df):
    # Clean df
    fresh_df.drop('Stock Splits', axis=1, inplace=True)
    fresh_df.drop('Dividends', axis=1, inplace=True)

    # Calculate Returns and append to the df DataFrame
    fresh_df.ta.percent_return(cumulative=False, append=True)
    fresh_df["up"] = (fresh_df.ta.percent_return(cumulative=False) > 0)


    #  Next close
    lastCloseDf =fresh_df.tail(1)
    lastClose = lastCloseDf.iloc[0]['Close']

    fresh_df['Next Close'] = fresh_df['Close'].shift(-1, fill_value=lastClose)

    fresh_df.ta.inertia(append=True)
    fresh_df.ta.rsi(append=True)
    fresh_df.ta.vwap(append=True)
    fresh_df.ta.cdl_pattern(name=["doji"],append=True)
    fresh_df['SMA 10'] = fresh_df.ta.sma(10)
    fresh_df['SMA 50'] = fresh_df.ta.sma(50)
    fresh_df['SMA 200'] = fresh_df.ta.sma(200)
    fresh_df['EMA 20'] = fresh_df.ta.ema(20)
    fresh_df['GoldenCross'] = (fresh_df['SMA 50'] > fresh_df['SMA 200'])
    fresh_df.ta.obv(append=True)

    return fresh_df


df = build_dataFrame(df)



def train_model(symbol):
    logging.info('Training model for %s',symbol)
    training_df = pd.DataFrame()
    training_df = training_df.ta.ticker(symbol, period="max", interval="1d")
    training_df = build_dataFrame(training_df)

    training_df=training_df.fillna(training_df.mean())

    X = training_df.drop('Next Close',axis=1)
    y = training_df['Next Close']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    model = RandomForestRegressor(n_estimators = 1000, random_state = 42)
    model.fit(X_train, y_train)

    # store model
    name = symbol + '-model.pkl'
    joblib.dump(model, 'models/'+name)

    y_pred = model.predict(X_test)

    result=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})


    # print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    # print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    # print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    # Mean Absolute Error: 1993.2901175839186 # <20%

    # Calculate the absolute errors
    errors = abs(y_pred - y_test)
    # Print out the mean absolute error (mae)
    # print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

    # Calculate mean absolute percentage error (MAPE)
    mape = 100 * (errors / y_test)
    # Calculate and display accuracy
    accuracy = 100 - np.mean(mape)
    print('Accuracy:', round(accuracy, 2), '%.')
    result.sort_index(inplace=True)
    
    return accuracy, model



# INFERENCE

# remove object fields from df
# df = df.select_dtypes(exclude=['object'])

# change NAN with mean value
df=df.fillna(df.mean())

X = df.drop('Next Close',axis=1)
y = df['Next Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

model = RandomForestRegressor(n_estimators = 1000, random_state = 42)
model.fit(X_train, y_train)

# store model
name = SYMBOL + '-model.pkl'
# joblib.dump(model, 'models/'+name)

y_pred = model.predict(X_test)

result=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# Mean Absolute Error: 1993.2901175839186 # <20%

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
# print(result.tail(5))
tail = result.tail(155)
# tail.plot(figsize=(10, 4))



# ====================
# predict today/tomorrows close
# ====================

# today = True
# index = 0
# today_df = df.tail(2)
# today_df = today_df.drop('Next Close',axis=1)
# print(today_df)

# pred = model.predict(today_df)
# print('tomorrows close prediction:', pred)

# if today:
#     print('getting todays close...')
#     index = 0
# else:
#     print('getting tomorrows close...')
#     index = 1

# todayClose = today_df.iloc[index]['Close']
# pred = pred[index]
# print('Predicted Close',pred)


# diff = pred - todayClose
# diff_percent = (diff/todayClose)*100
# print('Predicted %', round(diff_percent, 3))



# print Columns
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
        self.sleeptime = "1H"
        self.last_trade = None 
        self.cash_at_risk = cash_at_risk
        self.threshold = 1
        self.trail_percent = 2.5
        self.minutes_before_closing = 5
        self.period = '1y'

        if self.is_backtesting:
            print("Running in backtesting mode")
            self.sleeptime = "24H"
            self.period = '5y'

    def position_sizing(self): 
        cash = self.get_cash() 
        last_price = self.get_last_price(self.symbol)
        quantity = round(cash * self.cash_at_risk / last_price,0)
        return cash, last_price, quantity

    def get_predicted_close(self):
        today = self.get_datetime()
        todayStr = today.strftime("%Y-%m-%d")
        print('Date', todayStr)

        if self.is_backtesting:
            newData = df
        else:
            newData = pd.DataFrame() # Empty DataFrame
            newData = newData.ta.ticker(SYMBOL, period=self.period, interval="1d")
            # print(newData.tail(5))
            newData = build_dataFrame(newData)

        
        today_df = newData[todayStr:todayStr]
        today_df = today_df.drop('Next Close',axis=1)

        if today_df.empty:
            print('*****************')
            print('No data for', todayStr)
            print('*****************')
            return 0

        # print(today_df)
        pred = model.predict(today_df)
        print(pred)
        return pred
    
    def before_market_closes(self):
        print('Before market close event!')
        self.on_trading_iteration()

    def on_trading_iteration(self):

        if self.first_iteration:
            print('first iteration')
            self.await_market_to_close(5) # pause trading_iteration until 5mins before close

        cash, last_price, quantity = self.position_sizing() 
        pred_close = self.get_predicted_close()

        if pred_close == 0:
            pred_close = last_price

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

start_date = datetime(2020,8,3)
end_date = datetime(2024,8,19) 
broker = Alpaca(ALPACA_CREDS) 
strategy = MLTrader(name='mlstrat', broker=broker, 
                    parameters={"symbol":SYMBOL, 
                                "cash_at_risk":.8})




if __name__ == '__main__':
    print(os.environ.get('APP_FILE'))
    if os.environ.get('APP_FILE'):
        trader = Trader()
        trader.add_strategy(strategy)
        trader.run_all()
    else:
        # print('helo')
        strategy.backtest(
            YahooDataBacktesting, 
            start_date, 
            end_date, 
            parameters={"symbol":SYMBOL, "cash_at_risk":.8},
            benchmark_asset=SYMBOL
        )

