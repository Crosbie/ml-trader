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
        accuracy, model = train_model(symbol)
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

            x = ' '.join(("multiline String ",
            "Python Language",
            "Welcome to GFG"))

            data = ' '.join(('<h4>'+symbol+'</h4>',
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
    

@app.route('/predict/<int:index>', methods=['GET'])
def predict(index):
    if model1:
        try:
            # json_ = request.json
            # print(json_)
            # query = pd.get_dummies(pd.DataFrame(json_))
            # query = query.reindex(columns=model_columns, fill_value=0)

            AAPL_df = getData('AAPL').tail(2)
            AAPL_df = AAPL_df.drop('Next Close',axis=1)

            GOLD_df = getData('GC=F').tail(2)
            GOLD_df = GOLD_df.drop('Next Close',axis=1)
            

            # prediction = list(model.predict(query))

            pred1 = model1.predict(AAPL_df)
            pred2 = model2.predict(GOLD_df)

            msg1 = "AAPL: Todays Close prediction"
            msg2 = "GOLD: Todays Close prediction"

            if index is None:
                index = 1

            if index == 1:
                msg1 = "AAPL: Tomorrows Close prediction"
                msg2 = "GOLD: Tomorrows Close prediction"

            return jsonify({msg1: str(pred1[index]), msg2: str(pred2[index])})
            

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')
    
def getData(SYMBOL):
    df = pd.DataFrame() # Empty DataFrame
    df = df.ta.ticker(SYMBOL, period="1y", interval="1d")
    df = build_dataFrame(df)
    return df

if __name__ == '__main__':
    port = os.environ.get('FLASK_PORT') or 8080
    port = int(port)

    model1 = joblib.load("models/AAPL-model.pkl") # Load "model.pkl"
    model2 = joblib.load("models/GC=F-model.pkl") # Load "model.pkl"
    print ('Model loaded')
    # model_columns = joblib.load("model_columns.pkl") # Load "model_columns.pkl"
    print ('Model columns loaded')

    app.run(port=port,host='0.0.0.0')

    # app.run(port=port, debug=True)