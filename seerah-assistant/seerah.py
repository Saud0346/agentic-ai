import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PDFPlumberLoader

# ------------------------
# 1. Load API Key
# ------------------------
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

# ------------------------
# 2. Load Documents (PDFs)
# ------------------------
pdf_path = "data/raheeq.pdf"   # replace with your own file
loader = PDFPlumberLoader(pdf_path)
documents = loader.load()

# ------------------------
# 3. Split Documents into Chunks
# ------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
docs = text_splitter.split_documents(documents)

# ------------------------
# 4. Create Vector DB (FAISS)
# ------------------------
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = FAISS.from_documents(docs, embeddings)

# ------------------------
# 5. Build Retrieval-QA Chain
# ------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",   # fast + cost-efficient model
    google_api_key=api_key,
    temperature=0
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type="stuff"   # simplest method
)

# ------------------------
# 6. Ask Questions
# ------------------------
while True:
    query = input("\nAsk a question (or 'exit'): ")
    if query.lower() == "exit":
        break
    result = qa_chain.run(query)
    print(f"\nAnswer: {result}")