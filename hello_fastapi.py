from fastapi import FastAPI 
from pydantic import BaseModel
from typing import Optional
import pandas as pd 

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

@app.post('/user-data')
def user_data(user_info: UserData):
    return f'His/Her name is {user_info.name} and surnames {user_info.surname}, {user_info.second_surname}'

@app.get('/get-dataframe')
def read_dataframe_properties(): 
    df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')
    return df 