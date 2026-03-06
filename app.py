import streamlit as st
import pdfplumber
import io
import json
import google.genai as genai
from google.genai import types
from typing import Any
import requests

# ------------------- Page Setup -------------------
st.set_page_config(page_title="AI Powered Document Orchestrator", layout="wide")
st.title("AI Powered Document Orchestrator")
st.caption("Upload a PDF or TXT document to extract content and perform operations.")

# ------------------- Helper Functions -------------------
def get_secret(key):
    value = st.secrets.get(key)
    if not value:
        st.warning(f"Secret '{key}' is not set. Please add it in Streamlit secrets.")
        return None
    return value

def extract_text_from_pdf(file):
    if file is None:
        st.warning("Please upload a PDF or TXT file to extract text.")
        return ""
    
    file_name = file.name.lower()
    if file_name.endswith('.txt'):
        return file.getvalue().decode('utf-8')
    elif file_name.endswith('.pdf'):
        content = file.getvalue()
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            pages = [page.extract_text() for page in pdf.pages]
            return "\n\n".join(pages)
    else:
        st.warning("Unsupported file type. Please upload a PDF or TXT file.")
        return ""

def normalize_json_payload(text):
    raw = text.strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {str(e)}")
    
    if isinstance(parsed, dict) and 'key_points' in parsed:
        return parsed
    
    if isinstance(parsed, dict):
        key_points = [{'key': k, 'value': v, 'relevance': 'Extracted'} for k, v in parsed.items()]
        return {"key_points": key_points}

    raise ValueError("Invalid JSON format. Expected a dictionary or an object with 'key_points' key.")

def extract_structured_data(client: genai.Client, text: str, question: str) -> dict[str, Any]:
    schema = {
        "type": "object",
        "properties": {
            "key_points": {
                "type": "array",
                "minItems": 5,
                "maxItems": 8,
                "items": {
                    "type": "object",
                    "properties": {
                        "key": {"type": "string"},
                        "value": {"type": "string"},
                        "relevance": {"type": "string"},
                    },
                    "required": ["key", "value", "relevance"],
                },
            }
        },
        "required": ["key_points"],
    }

    prompt = f"""
You are a precise document intelligence system.
Analyze the provided document and user question.
Extract ONLY the 5-8 most relevant key-value pairs required to answer the question.
Each item must include:
- key: concise field name
- value: extracted or inferred value from the document
- relevance: why this field matters for the question

Return valid JSON only, matching schema.

User Question:
{question}

Document Text:
{text[:20000]}
"""

    response = client.models.generate_content(
        model="models/gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
            temperature=0.2,
        ),
    )

    return normalize_json_payload(response.text)

def call_n8n_webhook(
    webhook_url: str,
    full_text: str,
    question: str,
    structured_data: dict[str, Any],
    recipient_email: str,
) -> dict[str, Any]:
    payload = {
        "document_text": full_text,
        "question": question,
        "structured_data": structured_data,
        "recipient_email": recipient_email,
    }

    response = requests.post(webhook_url, json=payload, timeout=120)
    response.raise_for_status()
    try:
        return response.json()
    except ValueError:
        return {
            "final_answer": response.text,
            "generated_email_body": "No separate email body returned.",
            "email_automation_status": "UNKNOWN",
        }

# ------------------- Session State -------------------
if "structured_data" not in st.session_state:
    st.session_state.structured_data = None
if "doc_text" not in st.session_state:
    st.session_state.doc_text = ""
if "webhook_result" not in st.session_state:
    st.session_state.webhook_result = None

# ------------------- User Inputs -------------------
uploaded_files = st.file_uploader("Upload PDF or TXT files", type=["pdf", "txt"])
user_question = st.text_input("Enter your question about the document:")

# ------------------- Extract Structured Data -------------------
if st.button('Extract Structured Data'):
    if not uploaded_files:
        st.warning("Please upload a PDF or TXT file to extract data.")
    elif not user_question:
        st.warning("Please enter a question to guide the data extraction process.")
    else:
        gemini_api_key = get_secret("GEMINI_API_KEY")
        if gemini_api_key:
            with st.spinner('Extracting text and querying AI...'):
                try:
                    # Single file support
                    extracted_text = extract_text_from_pdf(uploaded_files)
                    if not extracted_text.strip():
                        st.warning("No readable text found in the uploaded file.")
                    else:
                        client = genai.Client(api_key=gemini_api_key)
                        structured_data = extract_structured_data(client, extracted_text, user_question)
                        st.session_state.doc_text = extracted_text
                        st.session_state.structured_data = structured_data
                        st.success("Structured data extracted successfully!")
                except Exception as e:
                    st.error(f"An error occurred during extraction: {str(e)}")

# ------------------- Display Structured Data -------------------
if st.session_state.structured_data:
    st.subheader("Extracted Structured Data")
    st.json(st.session_state.structured_data)
    key_points = st.session_state.structured_data.get("key_points", [])
    if key_points:
        st.dataframe(key_points, use_container_width=True)

    st.markdown("---")
    st.subheader("Trigger Conditional Email Automation (n8n)")
    recipient_email = st.text_input("Recipient Email ID", key="recipient_email_field")

    if st.button("Send Alert Mail"):
        n8n_webhook_url = get_secret("N8N_WEBHOOK_URL")
        if not recipient_email.strip():
            st.warning("Please enter a recipient email ID.")
        elif not n8n_webhook_url:
            st.warning("Missing N8N_WEBHOOK_URL in secrets.")
        else:
            with st.spinner("Calling n8n webhook..."):
                try:
                    webhook_result = call_n8n_webhook(
                        webhook_url=n8n_webhook_url,
                        full_text=st.session_state.doc_text,
                        question=user_question,
                        structured_data=st.session_state.structured_data,
                        recipient_email=recipient_email.strip(),
                    )
                    st.session_state.webhook_result = webhook_result
                    st.success("n8n workflow completed.")
                except Exception as exc:
                    st.exception(exc)

# ------------------- Display n8n Results -------------------
import re

if st.session_state.webhook_result:
    result = st.session_state.webhook_result  # Already dict
    
    st.markdown("---")
    st.subheader("Final Outputs")

    st.markdown("### Final Analytical Answer")
    # REGEX: Extract text between "final_answer": "..." or "final_answer": {...text...}
    final_answer_raw = str(result.get("final_answer", "{}"))
    
    # Method 1: Extract quoted string value
    match = re.search(r'"final_answer"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', final_answer_raw, re.DOTALL)
    if match:
        final_answer = match.group(1).replace('\\n', '\n').replace('\\"', '"')
    else:
        # Method 2: Extract text after "final_answer":
        match = re.search(r'"final_answer"\s*:\s*{[^}]*"([^"]{50,})"', final_answer_raw, re.DOTALL)
        final_answer = match.group(1) if match else final_answer_raw
    
    st.markdown(final_answer)

    st.markdown("### Generated Email Body")
    email_body = result.get("generated_email_body", "No email body returned.")
    if email_body == "Not Required":
        st.info(email_body)
    else:
        st.code(email_body, language="markdown")

    st.markdown("### Email Automation Status")
    status = result.get("email_automation_status", "UNKNOWN")
    if status == "Not Sent":
        st.info(status)    
    else:
        st.warning(status)

# ------------------- Deployment Note -------------------
st.markdown("---")
st.markdown(
    "Deployment note: configure `GEMINI_API_KEY` and `N8N_WEBHOOK_URL` in Streamlit secrets "
    "before running extraction and automation."
)