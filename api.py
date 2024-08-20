# Dependencies
from flask import Flask, request, jsonify
import joblib
import traceback
import pandas as pd
import numpy as np
from app import build_dataFrame

# Your API definition
app = Flask(__name__)

@app.route('/predict/<int:index>', methods=['GET'])
def predict(index):
    if model:
        try:
            # json_ = request.json
            # print(json_)
            # query = pd.get_dummies(pd.DataFrame(json_))
            # query = query.reindex(columns=model_columns, fill_value=0)

            today_df = getData('AAPL').tail(2)
            today_df = today_df.drop('Next Close',axis=1)
            print(today_df)

            # prediction = list(model.predict(query))

            pred = model.predict(today_df)

            msg = "AAPL: Todays Close prediction"

            if index == 1:
                msg = "AAPL: Tomorrows Close prediction"

            return jsonify({msg: str(pred[index])})
            

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
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345

    model = joblib.load("models/AAPL-model.pkl") # Load "model.pkl"
    print ('Model loaded')
    # model_columns = joblib.load("model_columns.pkl") # Load "model_columns.pkl"
    print ('Model columns loaded')

    app.run(port=port, debug=True)