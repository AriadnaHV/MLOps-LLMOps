from fastapi import FastAPI
import os
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
app = FastAPI()
# Inicializamos el modelo de LangChain
load_dotenv()
GOOGLE_API_KEY = os.getenv('GEMINI_API_KEY')

@app.get('/saluda')
def root():
  return {'Message': 'Hola soy Raquel'}


@app.get('/despido')
def root():
  return {'Message': 'Hola soy Raquel'}

@app.get('/expert')
def chain_invoke(tema, estilo, pregunta):
  llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.7
  ) 
  template = ChatPromptTemplate.from_messages([
      ("system", "Eres un experto en {tema}. Responde de forma {estilo}."),
      ("human", "{pregunta}")
  ])

  chain = template | llm | StrOutputParser()
  result = chain.invoke({
    "tema":  tema,
    "estilo": estilo,
    "pregunta": pregunta
  })
  return result

 

@app.get("/gemini-v2") 
def gemini_flash(query: str): 
  load_dotenv()
  genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

  # Create the model
  generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
  }

  model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    generation_config=generation_config,
  )

  chat_session = model.start_chat(
    history=[
    ]
  )

  response = chat_session.send_message(query)

  print(response.text)
  return response.text

@app.get("/gemini") 
def gemini_flash(query: str): 
  load_dotenv()
  genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

  # Create the model
  generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
  }

  model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
  )

  chat_session = model.start_chat(
    history=[
    ]
  )

  response = chat_session.send_message(query)

  print(response.text)
  return response.text