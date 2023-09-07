import os
import openai

from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.mapreduce import MapReduceChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain

# Load enviroment variables: OPENAI_API_KEY
load_dotenv()

# Access the API key using the variable name defined in the .env file
openai.api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(openai_api_key = openai.api_key)

def get_journals():
    text = ''
    for filename in os.listdir('Journals'):
        with open(os.path.join('Journals', filename),'r') as f:
            text += f.read()
            text += '\n'
    return text

journal_text = get_journals()

