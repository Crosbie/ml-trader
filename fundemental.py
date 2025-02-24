
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import yfinance as yf
from typing import Tuple 
from datetime import datetime
from timedelta import Timedelta
from alpaca_trade_api import REST
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


device = "cuda:0" if torch.cuda.is_available() else "cpu"

PANIC_KEY = "288ad280d5157ab6630dec7a4e94e636b575030d"

API_KEY = "PKAXVXWV2RHJ1MR5PVY5"
API_SECRET = "crdhBKc9AXRgwInzTgBf7Ds4bZ3FMd6GszjoluK4"
BASE_URL = "https://paper-api.alpaca.markets/v2"
api = REST(base_url=BASE_URL, key_id=API_KEY, secret_key=API_SECRET)

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)
labels = ["positive", "negative", "neutral"]

def estimate_sentiment(news):
    if news:
        tokens = tokenizer(news, return_tensors="pt", padding=True).to(device)

        result = model(tokens["input_ids"], attention_mask=tokens["attention_mask"])[
            "logits"
        ]
        result = torch.nn.functional.softmax(torch.sum(result, 0), dim=-1)
        probability = result[torch.argmax(result)]
        sentiment = labels[torch.argmax(result)]
        return probability, sentiment
    else:
        return 0, labels[-1]
    
# https://www.geeksforgeeks.org/python-sentiment-analysis-using-vader/
def get_vader_sentiment(news):
    if news:
        generalSentiment = 0
        newsLength = len(news)
        # Create a SentimentIntensityAnalyzer object.
        sid_obj = SentimentIntensityAnalyzer()

        for item in news:
            sentiment_dict = sid_obj.polarity_scores(news)
            generalSentiment = generalSentiment+sentiment_dict['compound']

        return (generalSentiment/newsLength)
    else:
        return 0
    
def get_dates(): 
    today = datetime.today()
    three_days_prior = today - Timedelta(days=3)
    return today.strftime('%Y-%m-%d'), three_days_prior.strftime('%Y-%m-%d')

def get_news(ticker):
    today, three_days_prior = get_dates()
    if ticker:
        # data = pd.DataFrame()
        # news = yf.Ticker(ticker).news
        # news = [ev["title"] for ev in news]
        # data["news"] = news

        news = api.get_news(symbol=ticker, 
            start=three_days_prior, 
            end=today) 
        news = [ev.__dict__["_raw"]["headline"] for ev in news]

        return news
    else:
        return []
    

def getSentiment():
    # need to implement
    print('Getting sentiment')


if __name__ == "__main__":
    # tensor, sentiment = estimate_sentiment(['markets responded negatively to the news!','traders were displeased!'])
    # print(tensor, sentiment)
    # print(torch.cuda.is_available())

    news = get_news('GME')
    print(news)

    print('')
    print('=========FinBert==============')
    
    tensor, sentiment = estimate_sentiment(news)
    print(tensor, sentiment)
    # print(torch.cuda.is_available())

    print('=========FinBert==============')
    print('')
    
    print('=========VADER==============')
    vsentiment = get_vader_sentiment(news)

    print(vsentiment)
    print('=========VADER==============')