from langchain.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.chains.question_answering import load_qa_chain
import os
from dotenv import load_dotenv
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
import google.generativeai as genai
genai.configure(api_key=google_api_key)
print("Available Gemini models:")
for m in genai.list_models():
	print(m)
import google.generativeai as genai
import google.generativeai as genai

# List available Gemini models for debugging
genai.configure(api_key=google_api_key)
for m in genai.list_models():
	print(m)

# Load API key from .env
from dotenv import load_dotenv
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=google_api_key)
print("Available Gemini models:")
for m in genai.list_models():
	print(m)

# List available Gemini models for debugging
genai.configure(api_key=google_api_key)
print("Available Gemini models:")
for m in genai.list_models():
	print(m)

# List available Gemini models for debugging
genai.configure(api_key=google_api_key)
print("Available Gemini models:")
for m in genai.list_models():
	print(m)

# Replace book.pdf with any pdf of your choice
loader = UnstructuredPDFLoader("book.pdf")
pages = loader.load_and_split()
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
docsearch = Chroma.from_documents(pages, embeddings).as_retriever()

# Choose any query of your choice
query = "Who is Rich Dad?"
docs = docsearch.get_relevant_documents(query)
content = "\n".join([x.page_content for x in docs])
qa_prompt = "Use the following pieces of context to answer the user's question. If you don't know the answer, just say that you don't know, don't try to make up an answer.\n----------------"
input_text = qa_prompt + "\nContext:" + content + "\nUser question:\n" + query
llm = ChatGoogleGenerativeAI(model="models/gemini-pro-latest", temperature=0, google_api_key=google_api_key)
result = llm.invoke(input_text)
	if isinstance(result.content, str):
		print(result.content.replace("\n", " "))
	elif isinstance(result.content, list):
		cleaned = " ".join([str(item).replace("\n", " ") for item in result.content])
		print(cleaned)
	else:
		print(str(result.content).replace("\n", " "))
