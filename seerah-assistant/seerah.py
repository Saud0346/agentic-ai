import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# --- Load API Key (for LLM, not embeddings) ---
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file")

# --- Load and process PDFs ---
docs = []
data_dir = "data"  # put your PDFs inside this folder
for file in os.listdir(data_dir):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(data_dir, file))
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs.extend(text_splitter.split_documents(documents))

# --- Use HuggingFace embeddings (local, free) ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- Create / Load Vector Store ---
if os.path.exists("vectorstore"):
    vectorstore = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
else:
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("vectorstore")

# --- Build Retriever + QA Chain ---
llm = ChatOpenAI(openai_api_key=api_key, model="gpt-4o-mini")  # you can also use gpt-3.5-turbo
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

# --- Simple Q&A Loop ---
print("\n📚 Seerah RAG Chatbot Ready! Ask your questions (type 'exit' to quit)\n")
while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        print("👋 Exiting chatbot.")
        break
    result = qa_chain({"query": query})
    print("\n🤖 Answer:", result["result"])
    print("\n🔎 Sources:")
    for doc in result["source_documents"]:
        print(" -", doc.metadata.get("source", "Unknown"))
    print("\n" + "-"*60 + "\n")
