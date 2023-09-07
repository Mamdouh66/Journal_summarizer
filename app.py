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
llm = ChatOpenAI(temperature=0.8, model_name="gpt-3.5-turbo-16k", openai_api_key = openai.api_key)

def get_journals():
    text = ''
    for filename in os.listdir('Journals'):
        with open(os.path.join('Journals', filename),'r') as f:
            text += f.read()
            text += '\n'
    return text


def creating_docs(textfile):
    text_splitter = RecursiveCharacterTextSplitter()
    docs = text_splitter.create_documents([textfile])
    print(f"number of docuemnts are {len(docs)} number of tokens in the first one is {llm.get_num_tokens(docs[0].page_content)}")
    return docs

def get_summary():
    map_prompt = """the following are daily journals, i want you to summarize them and analyze each day
                    feelings, highlights, lowlights, and what i learned, thoughs and ideas.
                    the text will be delimited by four backquotes.
                    ````{text}````
                    SUMMARY:"""
    
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])

    combine_prompt = """ write a concise summary about the following text delimited by four backquotes.
                        the text is my day journal and i want you to summarize how i felt today, it should be 3 sentences maximum
                        ````{text}```` 
                        JOURNAL SUMMARY:"""
    
    combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])

    chain = load_summarize_chain(llm=llm, chain_type="map_reduce", map_prompt=map_prompt_template, combine_prompt=combine_prompt_template)
    return chain

def main():
    text = get_journals()
    docs = creating_docs(text)
    chain = get_summary()
    print(chain.run(docs))

if __name__ == '__main__':
    main()