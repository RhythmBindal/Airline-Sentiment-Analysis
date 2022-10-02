import joblib
import re
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from fastapi import FastAPI

app = FastAPI()

@app.get('/')
def get_root():
    return {'message': 'Welcome to the sentiment analysis API'}

model = joblib.load('sentiment_analysis.joblib')
def preprocessor(text):
    text = re.sub('<[^>]*>', '', text) # Effectively removes HTML markup tags
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    return text
def classify_message(model, message):
    message = preprocessor(message)
    label = model.predict([message])[0]
    sentiment_prob = model.predict_proba([message])
    return {'label': label, 'sentiment_probability': sentiment_prob[0][1]}

@app.get('/sentiment_analysis_query/')
async def detect_sentiment_query(message: str):
	return classify_message(model, message)

@app.get('/sentiment_analysis_path/{message}')
async def detect_sentiment_path(message: str):
	return classify_message(model, message)