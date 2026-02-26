from fastapi import FastAPI 
from pydantic import BaseModel
from typing import Optional
import pandas as pd 
from transformers import pipeline
app = FastAPI()

class UserData(BaseModel): 
    name: str 
    surname: str 
    second_surname: Optional[str] = 'Valles'
    phone_number: str 

@app.get('/saluda')
def saludar(name): 
    return f'Hola soy {name}'

@app.get('/suma')
def suma(a: int, b: int) -> int: 
    return a + b

@app.post('/user-dataaaa')
def user_data(user_info: UserData):
    return f'His/Her name is {user_info.name} and surnames {user_info.surname}, {user_info.second_surname}'

@app.get('/get-dataframe/')
def read_dataframe_properties(position: int): 
    df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')
    value = df['sepal_length'][position]
    return value 

@app.get('/text-classification')
def text_classification():
    pipe = pipeline("text-classification")
    text_classified = pipe(["This restaurant is awesome", "Este restaurante es una basura", "This restaurant sucks", "This restaurant is awesome", "This restaurant is awesome"] )
    return text_classified
@app.get('/generate2')
def sentiment_classification(prompt):
  sentiment_pipeline = pipeline('summarization')
  return {'Sentiment': sentiment_pipeline(prompt)}