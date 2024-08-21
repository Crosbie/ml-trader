# Dependencies
import os
from flask import Flask, request, jsonify
import joblib
import traceback
import pandas as pd
import numpy as np
from app import build_dataFrame

# Your API definition
app = Flask(__name__)

@app.route('/')
def hello():
    return "Hello World!"
    

@app.route('/predict/<int:index>', methods=['GET'])
def predict(index):
    if model:
        try:
            # json_ = request.json
            # print(json_)
            # query = pd.get_dummies(pd.DataFrame(json_))
            # query = query.reindex(columns=model_columns, fill_value=0)

            AAPL_df = getData('AAPL').tail(2)
            AAPL_df = AAPL_df.drop('Next Close',axis=1)

            GOLD_df = getData('AAPL').tail(2)
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

            return jsonify({msg1: str(pred1[index],msg2: str(pred2[index])})
            

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