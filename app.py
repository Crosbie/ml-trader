import os
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import joblib
import logging
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
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

symbols = ['AAPL','^GSPC','BTC-USD', 'ETH-USD', '^GDAXI','GC=F','EURUSD=X','USDJPY=X','NVDA']
SYMBOL = symbols[3]
# SYMBOL = 'SOL-USD'
print('Symbol:',SYMBOL)

# Load data

# Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
# Default: "max"

# History by Interval by interval (including intraday if period < 60 days)
# Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
# Default: "1d"
df = df.ta.ticker(SYMBOL, period="5y", interval="1d")




def build_dataFrame1(fresh_df):
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
    fresh_df['Next_Dir'] = fresh_df['Next Close'] - fresh_df['Close']
    fresh_df.loc[fresh_df['Next_Dir'] > 0, 'Next Dir'] = 1
    fresh_df.loc[fresh_df['Next_Dir'] <= 0, 'Next Dir'] = 0

    # fresh_df['Next Dir'] = np.where(fresh_df['Next_Dir'] > 0,'long', 'short')
    # fresh_df['Next Dir'] = fresh_df['Next_Dir'] > 0


    fresh_df.drop('Next_Dir', axis=1, inplace=True)
    

    fresh_df.ta.inertia(append=True)
    fresh_df.ta.rsi(append=True)
    fresh_df.ta.ao(append=True)
    fresh_df.ta.macd(append=True)
    fresh_df.ta.mad(append=True)
    fresh_df.ta.adx(append=True)
    fresh_df.ta.ttm_trend(append=True)
    
    fresh_df.ta.vwap(append=True)

    # fresh_df['SMA 10'] = fresh_df.ta.sma(10)
    fresh_df['SMA 50'] = fresh_df.ta.sma(50)
    fresh_df['SMA 200'] = fresh_df.ta.sma(200)
    fresh_df['EMA 20'] = fresh_df.ta.ema(20)
    fresh_df['GoldenCross'] = (fresh_df['SMA 50'] > fresh_df['SMA 200'])
    fresh_df.ta.obv(append=True)

    return fresh_df

def build_dataFrame2(fresh_df):
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
    fresh_df['Next_Dir'] = fresh_df['Next Close'] - fresh_df['Close']
    fresh_df.loc[fresh_df['Next_Dir'] > 0, 'Next Dir'] = 1
    fresh_df.loc[fresh_df['Next_Dir'] <= 0, 'Next Dir'] = 0

    # fresh_df['Next Dir'] = np.where(fresh_df['Next_Dir'] > 0,'long', 'short')
    # fresh_df['Next Dir'] = fresh_df['Next_Dir'] > 0


    fresh_df.drop('Next_Dir', axis=1, inplace=True)
    

    fresh_df.ta.inertia(append=True)
    fresh_df.ta.rsi(append=True)
    fresh_df.ta.ao(append=True)
    fresh_df.ta.macd(append=True)
    fresh_df.ta.mad(append=True)
    fresh_df.ta.adx(append=True)
    fresh_df.ta.ttm_trend(append=True)
    
    fresh_df.ta.vwap(append=True)
    fresh_df.ta.cdl_pattern(name="all",append=True)

    # custom trends
    weekly_mean = fresh_df.rolling(7).mean()
    quarterly_mean = fresh_df.rolling(90).mean()
    annual_mean = fresh_df.rolling(365).mean()
    weekly_trend = fresh_df.shift(1).rolling(7).mean()["Next Dir"]

    # fresh_df["weekly_mean"] = weekly_mean["Close"] / fresh_df["Close"]
    # fresh_df["quarterly_mean"] = quarterly_mean["Close"] / fresh_df["Close"]
    # fresh_df["annual_mean"] = annual_mean["Close"] / fresh_df["Close"]

    # fresh_df["annual_weekly_mean"] = fresh_df["annual_mean"] / fresh_df["weekly_mean"]
    # fresh_df["annual_quarterly_mean"] = fresh_df["annual_mean"] / fresh_df["quarterly_mean"]
    fresh_df["weekly_trend"] = weekly_trend

    # fresh_df['SMA 10'] = fresh_df.ta.sma(10)
    fresh_df['SMA 50'] = fresh_df.ta.sma(50)
    fresh_df['SMA 200'] = fresh_df.ta.sma(200)
    fresh_df['EMA 20'] = fresh_df.ta.ema(20)
    fresh_df['GoldenCross'] = (fresh_df['SMA 50'] > fresh_df['SMA 200'])
    fresh_df.ta.obv(append=True)

    return fresh_df


df = build_dataFrame2(df)


# =======================
# Train Models
# model 1 is price
# model 2 is direction
# =======================

def train_model(symbol,period):
    if period is None:
        period = "5y"

    logging.info('Training model for %s over %s',symbol, period)
    training_df1 = pd.DataFrame()
    training_df1 = training_df1.ta.ticker(symbol, period=period, interval="1d")
    training_df1 = build_dataFrame1(training_df1)

    training_df2 = pd.DataFrame()
    training_df2 = training_df2.ta.ticker(symbol, period=period, interval="1d")
    training_df2 = build_dataFrame2(training_df2)

    training_df1=training_df1.fillna(training_df1.mean())
    training_df2=training_df2.fillna(training_df2.mean())


    X1 = training_df1.drop('Next Close',axis=1)
    y1 = training_df1['Next Close']
    X1 = X1.drop('Next Dir',axis=1)

    X2 = training_df2.drop('Next Close',axis=1)
    y2 = training_df2['Next Dir']
    X2 = X2.drop('Next Dir',axis=1)

    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=0)
    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=0, stratify=y2)

    # print(X1.tail(5))
    model1 = RandomForestRegressor(n_estimators = 2000, random_state = 42, max_features=0.8)
    model1.fit(X1_train, y1_train)

    # model2 = RandomForestClassifier(n_estimators = 1400, random_state = 42, max_features=0.3)
    model2 = RandomForestClassifier(n_estimators = 2200, min_samples_split=2, min_samples_leaf=4, max_depth=70, random_state = 42, max_features=0.8)
    model2.fit(X2_train, y2_train)
    prob = model2.predict_proba(X2_test)

    y1_pred = model1.predict(X1_test)
    y2_pred = model2.predict(X2_test)

    # print('')
    # print('prob',prob[-10:])
    # print('y2_pred',y2_pred[-10:])


    print('')
    print('===============================')
    print('          Performance          ',symbol);
    print('===============================')
    print('Model 1 R-score :', metrics.r2_score(y1_test, y1_pred))
    print('Model 2 R-score :', metrics.r2_score(y2_test, y2_pred))
    print('Model 2 Correct %: ', metrics.accuracy_score(y2_test, y2_pred, normalize=True)*100.0)
    print('')
    print('Matrix, True/False') 
    print(metrics.confusion_matrix(y2_test, y2_pred,labels=[1, 0]))
    accuracy = metrics.accuracy_score(y2_test, y2_pred)
    # precision = metrics.precision_score(y2_test, y2_pred)
    # recall = metrics.recall_score(y2_test, y2_pred)

    # print("Accuracy:", accuracy)
    # print("Precision:", precision)
    # print("Recall:", recall)
    print('===============================')
    print('')

    # hypertune(X1_train, y1_train,X1_test,y1_test)
    # hypertune2(X2_train, y2_train,X2_test,y2_test)


    # store model
    name1 = symbol + '-model.pkl'
    name2 = symbol + '-2-model.pkl'
    # joblib.dump(model1, 'models/'+name1)
    joblib.dump(model2, 'models/'+name2)

    

    result=pd.DataFrame({'Actual':y1_test, 'Predicted':y1_pred})


    # print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    # print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    # print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    # Mean Absolute Error: 1993.2901175839186 # <20%

    # Calculate the absolute errors
    errors = abs(y2_pred - y2_test)
    # Print out the mean absolute error (mae)
    # print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
    # print('pred',y2_pred)
    # print('target',y2_test)
    # print('\n\nerrors',errors)
    # Calculate mean absolute percentage error (MAPE)
    mape = 100 * (errors / y2_test)
    mape = mape.fillna(0)
    mape = mape.replace(np.inf, 100)
    # print('mape',mape)
    # Calculate and display accuracy
    accuracy = 100 - np.mean(mape)
    # print('Accuracy:', round(accuracy, 2), '%.')
    result.sort_index(inplace=True)
    
    return accuracy, model1, model2



# INFERENCE


def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * (errors / test_labels)
    mape = mape.fillna(0)
    mape = mape.replace(np.inf, 100)
    accuracy = 100 - np.mean(mape)
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy


def hypertune(x,y,xtest,ytest):
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 800, stop = 2200, num = 8)]
    # Number of features to consider at every split
    max_features = ['log2', 'sqrt',0.2,0.4,0.6,0.8]
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}
    print(random_grid)

    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    # rf = RandomForestRegressor()
    rf = RandomForestRegressor()
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 200, cv = 3, random_state=42, n_jobs = -1)
    # Fit the random search model
    rf_random.fit(x, y)

    # Look at parameters used by our current forest
    print('Parameters currently in use:\n')
    print(rf.get_params())

    print('')
    print('Best Params:')
    print(rf_random.best_params_)

    
    base_model = RandomForestRegressor(n_estimators = 1400, random_state = 42, max_features=0.3)
    base_model.fit(x, y)
    base_accuracy = evaluate(base_model, xtest, ytest)

    best_random = rf_random.best_estimator_
    random_accuracy = evaluate(best_random, xtest, ytest)

    print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))


def hypertune2(x,y,xtest,ytest):
    # Create the parameter grid based on the results of random search 
    # param_grid = {
    #     'bootstrap': [True],
    #     'max_depth': [20, 70, 110],
    #     'max_features': ['sqrt',0.2,0.8],
    #     'min_samples_leaf': [4],
    #     'min_samples_split': [2, 5, 10],
    #     'n_estimators': [1400, 1600, 2200]
    # }
    param_grid = {
        'bootstrap': [True],
        'max_depth': [20, 70, 110],
        'max_features': [0.8],
        'min_samples_leaf': [4,6],
        'min_samples_split': [10,20],
        'n_estimators': [2000, 2200]
    }
    

    # Create a based model
    # rf = RandomForestRegressor()
    rf = RandomForestClassifier()
    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 3, n_jobs = -1)
    # Fit the grid search to the data
    grid_search.fit(x, y)

    print('')
    print('Best Params:')
    print(grid_search.best_params_)

    
    base_model = RandomForestClassifier(n_estimators = 2200, min_samples_split=2, min_samples_leaf=4, max_depth=70, bootstrap=True, random_state = 42, max_features=0.8)
    base_model.fit(x, y)
    base_accuracy = evaluate(base_model, xtest, ytest)

    best_grid = grid_search.best_estimator_
    grid_accuracy = evaluate(best_grid, xtest, ytest)

    print('Improvement of {:0.2f}%.'.format( 100 * (grid_accuracy - base_accuracy) / base_accuracy))



# ==================================
#  Trade Strat
# ==================================


# from lumibot.brokers import Alpaca
# from lumibot.backtesting import YahooDataBacktesting
# from lumibot.strategies.strategy import Strategy
# from lumibot.traders import Trader
from datetime import datetime

class DittoBot(): 
    def initialize(self, symbol:str=SYMBOL, cash_at_risk:float=.5): 
        self.symbol = symbol
        self.sleeptime = "24H"
        self.last_trade = None 
        self.cash_at_risk = cash_at_risk
        self.threshold = 1
        self.trail_percent = 2.5
        self.minutes_before_closing = 5
        self.period = '1y'
        self.set_market("24/7")

        # base = "BTC"
        # quote = "USDT"
        # last_price = self.get_last_price(base, quote=quote)
        # self.log_message(f"Last price for BTC/USDT is {last_price}")


        # default for stock/gold/crypto
        # threshold = 1
        # trail_percent = 2.5

        # default for BTC
        # threshold = 0.1
        # trail_percent = 0.5

        if self.is_backtesting:
            print("Running in backtesting mode")
            self.sleeptime = "24H"
            self.period = '5y'

    def position_sizing(self): 
        cash = self.get_cash() 
        last_price = self.get_last_price(self.symbol) or 50000
        # print('Last_price',last_price)
        # print('cash',cash)
        # print('CAR',self.cash_at_risk)
        quantity = round(cash * self.cash_at_risk / last_price,0)
        # quantity = 1
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
            newData = build_dataFrame2(newData)

        
        today_df = newData[todayStr:todayStr]
        today_df = today_df.drop('Next Close',axis=1)
        today_df = today_df.drop('Next Dir',axis=1)

        if today_df.empty:
            print('*****************')
            print('No data for', todayStr)
            print('*****************')
            return 0, 0

        # print(today_df)
        pred = model2.predict(today_df)
        prob = model2.predict_proba(today_df)
        print(pred)

        confidence = prob.max(axis=1)
        print('Max today')
        print(confidence)
        result = map(lambda x: round(x*100,1), confidence)
        result = list(result)
        print(result)

        return pred, result[0]
    
    def before_market_closes(self):
        print('Before market close event!')
        self.on_trading_iteration()

    def on_trading_iteration(self):

        if self.first_iteration:
            print('first iteration')
            self.await_market_to_close(5) # pause trading_iteration until 5mins before close

        cash, last_price, quantity = self.position_sizing() 
        pred_close, prob = self.get_predicted_close()

        # if pred_close == 0:
        #     pred_close = last_price

        # diff = pred_close - last_price
        # diff_percent = (diff/last_price)*100

        # print(f"Diff: {diff}! percent: {diff_percent}")
        # print(diff > 0 and diff_percent > self.threshold)
        # print(diff < 0 and diff_percent < (self.threshold*-1))

        if cash > last_price: 
            if pred_close == 1 and prob > 72: 
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
            elif pred_close == 0 and prob > 72: 
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



def backtest(data, model, start=1000, step=550):
    predictions = []
    # Loop over the dataset in increments
    for i in range(start, data.shape[0], step):
        # Split into train and test sets
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        
        testTarget = test["Next Dir"]
        test.drop('Next Dir', axis=1, inplace=True)
        test.drop('Next Close', axis=1, inplace=True)

        target = train["Next Dir"]
        train.drop('Next Dir', axis=1, inplace=True)
        train.drop('Next Close', axis=1, inplace=True)
        
        # Fit the random forest model
        model.fit(train, target)
        
        # Make predictions
        preds = model.predict_proba(test)[:,1]
        preds = pd.Series(preds, index=test.index)
        preds[preds > .55] = 1
        preds[preds<=.55] = 0
        
        # Combine predictions and test values
        combined = pd.concat({"Target": testTarget,"Predictions": preds}, axis=1)
        
        predictions.append(combined)
    
    return pd.concat(predictions)     



start_date = datetime(2024,8,1)
end_date = datetime(2024,10,1) 
# broker = Alpaca(ALPACA_CREDS) 
# strategy = DittoBot(name='DittoBot', broker=broker, 
#                     parameters={"symbol":'BTC/USD', 
#                                 "cash_at_risk":.8})

# print('here')
if __name__ == '__main__':
    print(os.environ.get('APP_FILE'))
    if os.environ.get('APP_FILE'):
        # trader = Trader()
        # trader.add_strategy(strategy)
        # trader.run_all()
        print('Done')
    else:
        accuracy, model1, model2 = train_model(SYMBOL,'5y')
        print('Done');

        # predictions = backtest(df,model2)
        # print(predictions["Predictions"].value_counts())
        # print('')
        # print(predictions["Target"].value_counts())

        # print(metrics.precision_score(predictions["Target"], predictions["Predictions"]))

        # strategy.backtest(
        #     YahooDataBacktesting, 
        #     start_date, 
        #     end_date, 
        #     parameters={"symbol":SYMBOL, "cash_at_risk":.8},
        #     benchmark_asset=SYMBOL
        # )
        
