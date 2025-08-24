import os
import re
import json
import pdfplumber
from docx import Document
import pandas as pd
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from phi.agent import Agent
from phi.model.openai import OpenAIChat
import streamlit as st

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# ------------ File Readers ------------ #
def read_pdf(file):
    with pdfplumber.open(file) as pdf:
        return "\n".join([page.extract_text() or '' for page in pdf.pages]).strip()

def read_docx(file):
    doc = Document(file)
    return "\n".join([para.text for para in doc.paragraphs]).strip()

def read_txt(file):
    return file.read().decode('utf-8').strip()

def read_csv(file):
    df = pd.read_csv(file)
    return df.to_string()

def read_excel(file):
    df = pd.read_excel(file)
    return df.to_string()

def read_json_file(file):
    return json.dumps(json.load(file), indent=2)

def read_html(file):
    soup = BeautifulSoup(file, 'lxml')
    return soup.get_text().strip()

def read_any_file(uploaded_file):
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    readers = {
        ".pdf": read_pdf,
        ".docx": read_docx,
        ".txt": read_txt,
        ".csv": read_csv,
        ".xlsx": read_excel,
        ".xls": read_excel,
        ".json": read_json_file,
        ".html": read_html
    }
    return readers.get(ext, lambda f: "Unsupported file extension.")(uploaded_file)

# ------------ JSON Fixer ------------ #
def fix_malformed_json(text):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE)
    text = text.replace("'", '"')
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return text.strip()

# ------------ Agents ------------ #
validator_agent = Agent(
    model=OpenAIChat(id="gpt-4o", api_key=api_key),
    instructions="""
You are a strict and professional resume validator.

You will receive raw resume text and must check if these required categories are present:

- Email  
- Phone Number    
- Education Level  
- Location  
- Nationality  
- Certifications  

For each category:
1. State if it's **Present ✅** or **Missing ❌**.
2. If anything is missing, ask the user to provide the missing info.
3. If all are present, say the resume is complete.

Also, if the input doesn't look like a resume at all (e.g. gibberish or errors), say it's unreadable.

Format cleanly.
""",
    markdown=False,
)

json_agent = Agent(
    model=OpenAIChat(id="gpt-4o", api_key=api_key),
    instructions="""
You are a strict JSON formatter.

Given raw resume text, return ONLY a JSON object with the following categories:
- Email
- Phone Number
- Education Level
- Location
- Nationality
- Certifications

Return valid JSON only.
""",
    markdown=False,
)

# ------------ Streamlit App (ChatGPT-style for missing info) ------------ #
st.title("📄 Resume Parser with Validator")

uploaded_file = st.file_uploader("Upload your resume", type=["pdf", "docx", "txt", "csv", "xlsx", "xls", "json", "html"])

# Initialize session state for chat-style input
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if uploaded_file:
    resume_text = read_any_file(uploaded_file)

    # Step 1: Validator Agent
    st.subheader("✅ Validator Agent Output")
    validator_output = validator_agent.run(resume_text).content
    st.markdown(validator_output)

    # Step 2: Handle Missing Fields
    if "Missing ❌" in validator_output:
        st.warning("Some required fields are missing. Please enter them below:")

        # ChatGPT-style input
        user_input = st.text_input("Enter missing info here (press Enter to submit):", key="user_input")

        if user_input:
            # Add user input to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            # Merge all user-provided info with resume
            additional_text = "\n".join([item["content"] for item in st.session_state.chat_history])
            updated_text = additional_text + "\n" + resume_text
            st.info("✅ Additional info added and merged.")

            # Step 3: JSON Agent
            st.subheader("📦 JSON Agent Output")
            json_response = json_agent.run(updated_text).content
            st.code(json_response, language="json")

            # Step 4: Fix JSON if needed
            try:
                structured_data = json.loads(json_response)
            except json.JSONDecodeError:
                st.warning("⚠️ Raw JSON is malformed. Attempting to fix...")
                try:
                    cleaned = fix_malformed_json(json_response)
                    structured_data = json.loads(cleaned)
                    st.success("✅ JSON fixed and parsed successfully.")
                except json.JSONDecodeError as e:
                    st.error(f"❌ Failed to parse JSON even after fix.\nError: {e}")
                    structured_data = None

            # Step 5: Show and Save JSON
            if structured_data:
                st.subheader("✅ Final Parsed JSON")
                st.json(structured_data)
                json_filename = f"parsed_{uploaded_file.name.rsplit('.', 1)[0]}.json"
                st.download_button("💾 Download JSON", json.dumps(structured_data, indent=2), file_name=json_filename)

    else:
        st.success("🎉 Resume is complete. Proceeding to final JSON conversion...")
        # ... (keep the rest exactly the same as your current code)
