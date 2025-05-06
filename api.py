# Dependencies
import os
from flask import Flask, request, jsonify
import joblib
import traceback
import pandas as pd
import pandas_ta as ta
import numpy as np
from dotenv import load_dotenv
import boto3 
from io import BytesIO 
import yfinance as yf
from app import build_dataFrame1, build_dataFrame2, train_model
import json
# from fundemental import getSentiment

latest_preds = {}

# Load Env vars
load_dotenv()

S3_BUCKET = os.environ.get('S3_BUCKET')

page = '''<div style="font-family:verdana;background:#cdcdcd;border-radius:20px;padding:150px;margin:20px">
    <h2><a href="/">Home</a></h2>
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

@app.route('/json/prices/<symbol>')
def json_prices(symbol):

    df1, df2 = getData(symbol,"1y",None)
    df2 = df2.tail(2)

    # index 1 = today, 0 = yesterday
    obj = df2.iloc[1].to_json(orient="index")
    obj = json.loads(obj)

    data = {
        "msg":"ok",
        "data": obj
    }
    return data

@app.route('/json/pricesHours/<hours>/<symbol>')
def json_pricesHours(hours,symbol):
    hours = int(hours)

    df1, df2 = getData(symbol,"3mo","1h")
    df2 = df2.tail(hours)

    # index 1 = today, 0 = yesterday
    obj = df2.to_json(orient="index")
    obj = json.loads(obj)

    data = {
        "msg":"ok",
        "data": obj
    }
    return data

@app.route('/json/pricesDays/<days>/<symbol>')
def json_pricesDays(days,symbol):
    days = int(days)

    df1, df2 = getData(symbol,"1y","1d")
    df2 = df2.tail(days)

    # index 1 = today, 0 = yesterday
    obj = df2.to_json(orient="index")
    obj = json.loads(obj)

    data = {
        "msg":"ok",
        "data": obj
    }
    return data


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

    df1, df2 = getData(symbol,None,None)
    if df1.empty:
        msg = {"msg":"Invalid ticker",
               "symbol":symbol}
        return msg
    else:
        accuracy, model1, model2 = train_model(symbol,period)
        accuracy = round(accuracy, 2)
        accuracy = str(accuracy) + '%'
        msg = {"msg":"ok",
               "symbol":symbol,
               "period":period,
               "accuracy":accuracy
               }
        return msg


# Pull model from s3 bucket and save locally
@app.route('/json/pull/<symbol>')
def json_pull(symbol):

    df1, df2 = getData(symbol,None,None)
    if df2.empty:
        msg = {"msg":"Invalid ticker",
               "symbol":symbol}
        return msg
    else:
        model2 = read_model(S3_BUCKET+"/"+symbol+"-2-model.pkl")
        name2 = symbol + '-2-model.pkl'

        if model2 is None:
            return {"msg":"No model exists",
                    "model": name2}
        
        joblib.dump(model2, 'models/'+name2)
        msg = {"msg":"ok",
               "model": name2}
        return msg

@app.route('/json/fetch/<symbol>')
def json_fetch(symbol):

    df1, df2 = getData(symbol,None,None)
    if df1.empty:
        msg = {"msg":"Invalid ticker",
               "symbol":symbol
        }
        return msg
    else:
        try:
            model2 = joblib.load("models/"+symbol+"-2-model.pkl") # Load "model.pkl"
            print("model2")
            # print(model2)
            df1, df2 = getData(symbol,None,None)
            df1 = df1.tail(2)
            df1 = df1.drop('Next Close',axis=1)
            df1 = df1.drop('Next Dir',axis=1)
            # pred = model1.predict(df1)

            df2 = df2.tail(2)
            df2 = df2.drop('Next Close',axis=1)
            df2 = df2.drop('Next Dir',axis=1)
            pred2 = model2.predict(df2)
            probability = model2.predict_proba(df2)
            probability2 = pd.Series(probability[:,1], index=df2.index)

            confidence = probability.max(axis=1)
            print('Max today')
            print(confidence)
            result = map(lambda x: round(x*100,1), confidence)
            result = list(result)
            print(result)

            yesterdayOpen = df1.iloc[0]['Open']
            yesterdayClose = df1.iloc[0]['Close']

            todayClose = round(df1.iloc[1]['Close'],4)
            todayOpen = round(df1.iloc[1]['Open'],4)

            # todayDiff = pred[0] - yesterdayClose
            # tomorrowDiff = pred[1] - todayClose
            todayDiff = 0
            tomorrowDiff = 0

            todayDiff_pc = round((todayDiff/yesterdayClose)*100,2)
            tomorrowDiff_pc = round((tomorrowDiff/todayClose)*100,2)

            msg = {
                "msg":"ok",
                "symbol": symbol,
                "yesterdayOpen": yesterdayOpen,
                "yesterdayClose": yesterdayClose,
                "todayOpen": todayOpen,
                "todayClosePred": 0,
                "todayDiff": todayDiff,
                "todayDiff_pc": todayDiff_pc,
                "todayDirection": pred2[0],
                "todayDirectionConfidence": result[0],
                "tomorrowOpen": todayClose,
                "tomorrowClosePred": 0,
                "tomorrowDiff": tomorrowDiff,
                "tomorrowDiff_pc": tomorrowDiff_pc,
                "tomorrowDirection": pred2[1],
                "tomorrowDirectionConfidence": result[1]
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

    df1, df2 = getData(symbol,None,None)
    if df1.empty:
        msg = "Invalid ticker: "+symbol

        return page + msg
    else:
        accuracy, model1, model2 = train_model(symbol,"5y")
        accuracy = round(accuracy, 2)
        accuracy = str(accuracy) + '%'
        return page + 'Trained model on: '+ symbol + '. Accuracy: '+ accuracy

@app.route('/fetch/<symbol>')
def fetch(symbol):

    df1, df2 = getData(symbol,None,None)
    if df1.empty:
        msg = "Invalid ticker: "+symbol
        return page + msg
    else:
        try:
            # model1 = joblib.load("models/"+symbol+"-model.pkl") # Load "model.pkl"
            model2 = joblib.load("models/"+symbol+"-2-model.pkl") # Load "model.pkl"
            df1, df2 = getData(symbol,None,None)
            # df1 = df1.tail(2)
            # df1 = df1.drop('Next Close',axis=1)
            # df1 = df1.drop('Next Dir',axis=1)
            # pred = model1.predict(df1)

            df2 = df2.tail(2)
            df2 = df2.drop('Next Close',axis=1)
            df2 = df2.drop('Next Dir',axis=1)
            pred2 = model2.predict(df2)
            probability = model2.predict_proba(df2)
            probability2 = pd.Series(probability[:,1], index=df2.index)

            print(probability2)

            print(probability)

            
            confidence = probability.max(axis=1)
            print('Max today')
            print(confidence)
            result = map(lambda x: round(x*100,1), confidence)
            result = list(result)
            print(result)

            yesterdayOpen = df1['Open'][0]
            yesterdayClose = df1['Close'][0]

            todayClose = round(df1['Close'][1],4)
            todayOpen = round(df1['Open'][1],4)

            # todayDiff = pred[0] - yesterdayClose
            # tomorrowDiff = pred[1] - todayClose

            # todayDiff_pc = round((todayDiff/yesterdayClose)*100,2)
            # tomorrowDiff_pc = round((tomorrowDiff/todayClose)*100,2)


            data = ' '.join(('<h4>'+symbol+'</h4>',
            '<h5>Direction today[0]:</h5>',
             str(pred2[0]) + ' (confidence: ' + str(result[0]) + '%)',
             '<h5>Direction tomorrow[1]:</h5>',
             str(pred2[1]) + ' (confidence: ' + str(result[1]) + '%)',
            '<h5>Yesterday Close:</h5>',
             str(yesterdayClose),
            '<h5>Todays Open:</h5>',
             str(todayOpen),
            # '<h5>Todays Close (pred):</h5>',
            #  str(round(pred[0],4)),
            # '<span>diff:',
            #  str(todayDiff_pc) + '%',
            '</span></hr></br>',
            '<h5>Tomorrow Open:</h5>',
             str(todayClose),
            # '<h5>Tomorrow Close (pred):</h5>',
            #  str(round(pred[1],4)),
            # '<span>diff:',
            #  str(tomorrowDiff_pc) + '%',
            '</span></hr></br>'))

            return page + str(data)
        except:
            print ('Train the model first')
            print(traceback.format_exc())
            return page + 'Train the model first. No model here to use'
            # return jsonify({'trace': traceback.format_exc()})


def read_model(path):

    print('fetching model',path)
    file = None;
    access_key = os.environ.get('AWS_ACCESS_KEY')
    secret_key = os.environ.get('AWS_SECRET_KEY')
        
    s3_bucket, s3_key = path.split('/')[2], path.split('/')[3:]
    s3_key = '/'.join(s3_key)
    
    try:
        with BytesIO() as f:
            boto3.client("s3", aws_access_key_id=access_key, aws_secret_access_key=secret_key).download_fileobj(Bucket=s3_bucket, Key=s3_key, Fileobj=f)
            f.seek(0)
            file = joblib.load(f)
    except:
        return file

    return file


def getData(SYMBOL,period,interval):
    if not period:
        period = "2y"

    if not interval:
        interval = "1d"

    df1 = pd.DataFrame() # Empty DataFrame
    df1 = yf.Ticker(SYMBOL).history(period=period, interval=interval)
    df1 = build_dataFrame1(df1)

    df2 = pd.DataFrame() # Empty DataFrame
    df2 = yf.Ticker(SYMBOL).history(period=period, interval=interval)
    df2 = build_dataFrame2(df2)
    return df1, df2

if __name__ == '__main__':
    port = os.environ.get('FLASK_PORT') or 8000
    port = int(port)

    app.run(port=port,host='0.0.0.0')

    # app.run(port=port, debug=True)