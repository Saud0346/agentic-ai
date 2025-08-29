# read_book.py

"""
Script to generate question-answer pairs from a PDF named `my.pdf` in the current directory.

Setup:
1. Create a `.env` file in the same folder with:
    OPENAI_API_KEY=your_openai_api_key_here

2. Install dependencies:
    pip install langchain-community openai python-dotenv tiktoken PyPDF2
"""

import os
import json
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader

# ------------------------------
# Load environment variables
# ------------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file")

# ------------------------------
# Function to load PDF content
# ------------------------------
def load_pdf(file_path: str) -> str:
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

# ------------------------------
# Function to split text into chunks
# ------------------------------
def split_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)

# ------------------------------
# Function to generate Q&A pairs
# ------------------------------
def generate_qa_pairs(chunk: str, model_name: str = "gpt-4") -> list:
    chat = ChatOpenAI(model_name=model_name, openai_api_key=api_key, temperature=0.7)

    prompt_template = """
    You are an AI tutor. Generate 3 high-quality question-answer pairs
    based on the following text. Each pair should be concise and clear.

    Text:
    {text}

    Format strictly as JSON list of objects:
    [
        {{"question": "Question1", "answer": "Answer1"}},
        {{"question": "Question2", "answer": "Answer2"}},
        {{"question": "Question3", "answer": "Answer3"}}
    ]
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)

    try:
        response = chat.invoke(prompt.format_messages(text=chunk))

        # Extract text safely
        if hasattr(response, "content"):
            content = response.content
        else:
            content = str(response)

        # Parse JSON safely
        qa_list = json.loads(content)
        return [{"user": qa["question"], "assistant": qa["answer"]} for qa in qa_list]
    except Exception as e:
        print(f"Error generating/parsing Q&A: {e}")
        return []

# ------------------------------
# Function to save QA pairs to JSONL
# ------------------------------
def save_to_jsonl(qa_pairs: list, output_file: str):
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            for pair in qa_pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")
        print(f"Saved {len(qa_pairs)} Q&A pairs to {output_file}")
    except Exception as e:
        print(f"Error saving JSONL: {e}")

# ------------------------------
# Main function
# ------------------------------
def main():
    pdf_file = "my.pdf"  # fixed name
    output_file = "qa_output.jsonl"  # fixed output file

    if not os.path.exists(pdf_file):
        print(f"File {pdf_file} not found in current directory.")
        return

    print(f"Loading {pdf_file}...")
    text = load_pdf(pdf_file)
    if not text.strip():
        print("No text found in PDF.")
        return

    print("Splitting text into chunks...")
    chunks = split_text(text)
    print(f"Total chunks: {len(chunks)}")

    all_qa_pairs = []
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}...")
        qa_pairs = generate_qa_pairs(chunk)
        all_qa_pairs.extend(qa_pairs)

    print(f"Saving Q&A pairs to {output_file}...")
    save_to_jsonl(all_qa_pairs, output_file)
    print("Done!")

# ------------------------------
# Run the script
# ------------------------------
if __name__ == "__main__":
    main()
