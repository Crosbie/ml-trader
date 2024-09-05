# Dependencies
import os
from flask import Flask, request, jsonify
import joblib
import traceback
import pandas as pd
import numpy as np
from app import build_dataFrame, train_model

latest_preds = {}

page = '''<div style="font-family:verdana;background:#cdcdcd;border-radius:20px;padding:150px;margin:20px">
    <h2><a href="/">Home</a></h2>
    <h2><a href="/train/AAPL">/train/(symbol)</a> to train a new model, takes <40sec</h2>
    <h2><a href="/fetch/AAPL">/fetch/(symbol)</a> to predict using model</h2>
    <h2><a href="/models">/models</a> to list all trained models</h2>
'''

# Your API definition
app = Flask(__name__)

@app.route('/')
def hello():
    return page


# =============================
# JSON Routes
# =============================

@app.route('/json/models')
def json_models():

    filenames = next(os.walk('models/'), (None, None, []))[2]  # [] if no file
    data = {
        "msg":"ok",
        "files": filenames
    }
    return data

@app.route('/json/train/<symbol>/<period>')
def json_train(symbol,period):

    df = getData(symbol)
    if df.empty:
        msg = {"msg":"Invalid ticker",
               "symbol":symbol}
        return msg
    else:
        accuracy, model = train_model(symbol,period)
        accuracy = round(accuracy, 2)
        accuracy = str(accuracy) + '%'
        msg = {"msg":"ok",
               "symbol":symbol,
               "period":period,
               "accuracy":accuracy
               }
        return msg


@app.route('/json/fetch/<symbol>')
def json_fetch(symbol):

    df = getData(symbol)
    if df.empty:
        msg = {"msg":"Invalid ticker",
               "symbol":symbol
        }
        return msg
    else:
        try:
            model = joblib.load("models/"+symbol+"-model.pkl") # Load "model.pkl"
            df = getData(symbol).tail(2)
            df = df.drop('Next Close',axis=1)
            pred = model.predict(df)


            yesterdayOpen = df.iloc[0]['Open']
            yesterdayClose = df.iloc[0]['Close']

            todayClose = round(df.iloc[1]['Close'],4)
            todayOpen = round(df.iloc[1]['Open'],4)

            todayDiff = pred[0] - yesterdayClose
            tomorrowDiff = pred[1] - todayClose

            todayDiff_pc = round((todayDiff/yesterdayClose)*100,2)
            tomorrowDiff_pc = round((tomorrowDiff/todayClose)*100,2)

            msg = {
                "msg":"ok",
                "symbol": symbol,
                "yesterdayOpen": yesterdayOpen,
                "yesterdayClose": yesterdayClose,
                "todayOpen": todayOpen,
                "todayClosePred": round(pred[0],4),
                "todayDiff": todayDiff,
                "todayDiff_pc": todayDiff_pc,
                "tomorrowOpen": todayClose,
                "tommorrowClosePred": round(pred[1],4),
                "tomorrowDiff": tomorrowDiff,
                "tomorrowDiff_pc": tomorrowDiff_pc
            }


            return msg
        except:

            # print ('Train the model first')
            print(traceback.format_exc())

            msg = {
                "msg":"Error reading model",
                "symbol": symbol,
                "trace": traceback.format_exc()
            }
            return msg
            # return jsonify({'trace': traceback.format_exc()})





# =============================
# Web Page Routes
# =============================

@app.route('/models')
def models():

    filenames = next(os.walk('models/'), (None, None, []))[2]  # [] if no file
    model_page = page + str(filenames)
    return model_page

@app.route('/train/<symbol>')
def train(symbol):

    df = getData(symbol)
    if df.empty:
        msg = "Invalid ticker: "+symbol

        return page + msg
    else:
        accuracy, model = train_model(symbol,"5y")
        accuracy = round(accuracy, 2)
        accuracy = str(accuracy) + '%'
        return page + 'Trained model on: '+ symbol + '. Accuracy: '+ accuracy

@app.route('/fetch/<symbol>')
def fetch(symbol):

    df = getData(symbol)
    if df.empty:
        msg = "Invalid ticker: "+symbol
        return page + msg
    else:
        try:
            model = joblib.load("models/"+symbol+"-model.pkl") # Load "model.pkl"
            df = getData(symbol).tail(2)
            df = df.drop('Next Close',axis=1)
            pred = model.predict(df)

            # data = {}
            # data[symbol] = [
            #     {"Todays Close": pred[0]},
            #     {"Tomorrows Close" :pred[1]}
            # ]

            yesterdayOpen = df['Open'][0]
            yesterdayClose = df['Close'][0]

            todayClose = round(df['Close'][1],4)
            todayOpen = round(df['Open'][1],4)

            todayDiff = pred[0] - yesterdayClose
            tomorrowDiff = pred[1] - todayClose

            todayDiff_pc = round((todayDiff/yesterdayClose)*100,2)
            tomorrowDiff_pc = round((tomorrowDiff/todayClose)*100,2)


            data = ' '.join(('<h4>'+symbol+'</h4>',
            '<h5>Yesterday Close:</h5>',
             str(yesterdayClose),
            '<h5>Todays Open:</h5>',
             str(todayOpen),
            '<h5>Todays Close (pred):</h5>',
             str(round(pred[0],4)),
            '<span>diff:',
             str(todayDiff_pc) + '%',
            '</span></hr></br>',
            '<h5>Tomorrow Open:</h5>',
             str(todayClose),
            '<h5>Tomorrow Close (pred):</h5>',
             str(round(pred[1],4)),
            '<span>diff:',
             str(tomorrowDiff_pc) + '%',
            '</span></hr></br>'))

            return page + str(data)
        except:
            print ('Train the model first')
            print(traceback.format_exc())
            return page + 'Train the model first. No model here to use'
            # return jsonify({'trace': traceback.format_exc()})
    
def getData(SYMBOL):
    df = pd.DataFrame() # Empty DataFrame
    df = df.ta.ticker(SYMBOL, period="1y", interval="1d")
    df = build_dataFrame(df)
    return df

if __name__ == '__main__':
    port = os.environ.get('FLASK_PORT') or 8080
    port = int(port)

    app.run(port=port,host='0.0.0.0')

    # app.run(port=port, debug=True)