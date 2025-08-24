import os
import pdfplumber
from docx import Document
import pandas as pd
import json
from google import genai
from dotenv import load_dotenv
from bs4 import BeautifulSoup

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# --- File Readers ---
def read_pdf(file_path):
    try:
        with pdfplumber.open(file_path) as pdf:
            text = "\n".join([page.extract_text() or '' for page in pdf.pages])
        return text.strip() if text.strip() else "No readable text found in PDF."
    except Exception as e:
        return f"Error reading PDF: {e}"

def read_docx(file_path):
    try:
        doc = Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text.strip() if text.strip() else "No readable text found in DOCX."
    except Exception as e:
        return f"Error reading DOCX: {e}"

def read_txt(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return text.strip() if text.strip() else "Empty text file."
    except Exception as e:
        return f"Error reading TXT: {e}"

def read_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        return df if not df.empty else "Empty CSV file."
    except Exception as e:
        return f"Error reading CSV: {e}"

def read_excel(file_path):
    try:
        df = pd.read_excel(file_path)
        return df if not df.empty else "Empty Excel file."
    except Exception as e:
        return f"Error reading Excel: {e}"

def read_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        return f"Error reading JSON: {e}"

def read_html(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'lxml')
        text = soup.get_text()
        return text.strip() if text.strip() else "No readable text found in HTML."
    except Exception as e:
        return f"Error reading HTML: {e}"

def read_any_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    readers = {
        ".pdf": read_pdf,
        ".docx": read_docx,
        ".txt": read_txt,
        ".csv": read_csv,
        ".xlsx": read_excel,
        ".xls": read_excel,
        ".json": read_json,
        ".html": read_html
    }
    return readers.get(ext, lambda x: f"Unsupported file extension: {ext}")(file_path)


# --- Step 1: Analyze & Validate Resume ---
def analyze_and_validate_resume(data):
    """
    Analyze resume text and validate categories.
    Returns human-readable ticks/crosses and missing prompts.
    """
    client = genai.Client(api_key=api_key)
    
    prompt = f"""
You are an expert Resume Analyzer and Validator.

Extract the following categories from the provided text:
- Email
- Phone Number
- Education Level
- Location
- Nationality
- Certifications
- Major Qualifications

Instructions:
1. For each category, return a tick ✅ if present, cross ❌ if missing.
2. If any category is missing, politely ask the user to provide the missing information.
3. If all categories are present, thank the user and confirm completion.
4. Keep your response concise and user-friendly.

Resume text:
{data}
"""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text


# --- Step 2: Convert to Structured JSON ---
def get_clean_resume_json(data):
    """
    Convert analyzed resume text into structured JSON.
    JSON is generated only if all categories are present.
    """
    import re

    def build_prompt(data):
        return f"""
Categorize each entry in JSON format:
- Email
- Phone Number
- Education Level
- Location
- Nationality
- Certifications
- Major Qualifications

Data:
\"\"\"
{data}
\"\"\"
"""

    def extract_json(text):
        try:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            return match.group() if match else None
        except:
            return None

    client = genai.Client(api_key=api_key)
    prompt = build_prompt(data)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    json_str = extract_json(response.text)

    if json_str:
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None
    else:
        return None