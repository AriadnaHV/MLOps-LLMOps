from fastapi import FastAPI
from pydantic import BaseModel  # PYdantic has other functionalities such as unifying Snake case and Camel case
from typing import Optional   # enables optional fields
import pandas as pd
from transformers import pipeline

# Connect with FastAPI everytime we call 

app = FastAPI()
# @app.get() or @app.post()

class UserData(BaseModel):  # Class UserData (child) inherits properties from parent BaseModel
    name: str
    surname: str
    second_surname: Optional[str]
    phone_number: str

# We need to "decorators" (they give functionality to our code)

@app.get("/saluda")
def saludar(name: str):
    return f"Hello, {name}. I hope your day is as beautiful as you are :)"

@app.get('/suma')
def sumar(arg1: float, arg2:float) -> float:
    return arg1 + arg2

@app.get('/percentage-of-a-number')
def calculate_percent(percent: float, value: float) -> str:
    answer = percent * value / 100
    return f"{percent}% of {value} = {round(answer,2)}"

@app.post('/user-data')
def user_data(user_info: UserData):
    return f'His/Her name is {user_info.name} and surnames {user_info.surname} {user_info.second_surname}'

@app.get('/get-dataframe')
def read_dataframe_properties(): 
    df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')
    return df 

@app.get('/get-sepal-length')
def read_sepal_length(position: int): 
    df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')
    value = df['sepal_length'][position]
    return value 

@app.get('/get-flower-dimensions')
def read_flower_dimensions(position: int) -> str: 
    df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')
    sepal_length = df['sepal_length'][position]
    sepal_width = df['sepal_width'][position]
    petal_length = df['petal_length'][position]
    return (f"Sepal length = {sepal_length}     Sepal width = {sepal_width}     Petal length = {petal_length}")

@app.get('/classify-text')
def classifier():
    classifier = pipeline("text-classification")
    text_classified = classifier(['Amazing park, with lots of weird creatures!',
                     'Nice setting, but the food was terrible',
                     'Not bad at all',
                     'This was a blast',
                     'If you care about your health, avoid at all costs'])
    return text_classified

@app.get('/analyze_sentiment')
def sentiment_analyzer(text):
    sentiment_analyzer = pipeline("sentiment-analysis")
    return sentiment_analyzer(text)

@app.get('/recognize-entities')
def entity_recognizer():
    ner_pipeline = pipeline("ner", 
                            model="dbmdz/bert-base-cased-finetuned-conll03-english", 
                            aggregation_strategy="simple")
    entities_recognized = ner_pipeline([
        'Both MIT and Harvard are excellent schools', 
        'The airline companies Aer Lingus and Level should team up'])
    return entities_recognized

@app.post('/summarize_text')
def text_summarization(text: str):
    summarizer = pipeline("summarization")
    summary = summarizer(text, max_length = 120, min_length = 20, do_sample = False)    
    return summary[0]['summary_text']

@app.get('/generate2')
def sentiment_classification(prompt):
    sentiment_pipeline = pipeline("summarization")
    return {'Sentiment': sentiment_pipeline(prompt)}
    