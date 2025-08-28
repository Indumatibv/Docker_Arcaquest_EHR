import json
import os
import logging
from dotenv import load_dotenv
from openai import OpenAI
from fastapi import FastAPI, Request, HTTPException
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
import psycopg2
from psycopg2.extras import RealDictCursor

# -----------------------
# Config
# -----------------------
logging.basicConfig(level=logging.INFO)
load_dotenv()

COLLECTION_NAME = "text_documents"

# -----------------------
# Generate summary
# -----------------------
def generate_summary(transcript_path: str, output_file: str):
    logging.info(f"Generating summary from {transcript_path}")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("❌ OPENAI_API_KEY not found in environment")

    client = OpenAI(api_key=api_key)

    with open(transcript_path, "r", encoding="utf-8") as f:
        full_text = f.read().strip()

    if not full_text:
        logging.warning("Transcript empty. No summary generated.")
        return ""

    prompt = f"""
    Summary:
    You are given a conversation between an interviewer and a participant. 
    The interviewer asks various questions about the participant's personal life, habits, diet, health, and daily routine.
    Write a detailed summary **only** from the participant's point of view using "I" statements...
    Transcript:
    {full_text}
    """

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1024
    )

    summary = response.choices[0].message.content.strip()
    with open(output_file, "w", encoding="utf-8") as out:
        out.write(summary)

    logging.info(f"✅ Summary saved to {output_file}")
    return summary

# -----------------------
# Build retriever
# -----------------------
def build_retriever(summary_file: str, db_connection: str):
    logging.info(f"Building vectorstore from {summary_file}")

    loader = TextLoader(summary_file, encoding="utf-8")
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(pages)

    embedding = OpenAIEmbeddings()
    vectorstore = PGVector.from_documents(
        documents=docs,
        embedding=embedding,
        collection_name=COLLECTION_NAME,
        connection_string=db_connection,
        pre_delete_collection=True  # automatically clears old embeddings
    )

    logging.info("✅ Retriever ready")
    return vectorstore.as_retriever()

# -----------------------
# Process fields
# -----------------------
def process_fields(dict_list, qa_chain):
    for dictionary in dict_list:
        if "options" in dictionary and isinstance(dictionary.get("value", ""), str):
            if dictionary["options"]:
                ehr_question = dictionary["label"]
                options = ",".join(dictionary["options"])
                question = f"Select the most appropriate answer from the given options only. USE ONLY THE PROVIDED CONTEXT TO ANSWER. Do NOT generate new options or free text. If the answer is not found, respond with 'No information found'. If multiple options are mentioned in the context, select the **first matching option** from the list below. Question: {ehr_question}, Options: {options}"
                
                dictionary["value"] = "No information found"  # reset default

                result = qa_chain.invoke(question)["result"]
                dictionary["value"] = result
                logging.info(f"Updated field '{ehr_question}' with value '{result}'")

        if "fields" in dictionary and isinstance(dictionary.get("value"), dict):
            for i, nested_dictionary in enumerate(dictionary["fields"]):
                if "options" in nested_dictionary:
                    ehr_question = nested_dictionary["label"]
                    options = ",".join(nested_dictionary["options"])
                    question = f"Select the most appropriate answer from the given options only. USE ONLY THE PROVIDED CONTEXT TO ANSWER. Do NOT generate new options or free text. If the answer is not found, respond with 'No information found'.If multiple options are mentioned in the context, select the **first matching option** from the list below. Question: {ehr_question}, Options: {options}"
                    
                    dictionary["value"]["fields"][i]["value"] = "No information found"  # reset default

                    result = qa_chain.invoke(question)["result"]
                    dictionary["value"]["fields"][i]["value"] = result
                    logging.info(f"Updated nested field '{ehr_question}' with value '{result}'")

# -----------------------
# FastAPI app
# -----------------------
app = FastAPI()

@app.post("/process-ehr-json/")
async def process_ehr_json(request: Request):
    logging.info("Received /process-ehr-json request")
    try:
        updated_json = await request.json()
    except Exception:
        logging.error("Invalid JSON body")
        raise HTTPException(status_code=400, detail="Invalid JSON body.")

    conversation = updated_json.get("conversation", [])
    if not conversation:
        logging.warning("No conversation found in JSON")
        raise HTTPException(status_code=400, detail="No conversation data found in JSON.")

    transcript_file = "ehr_transcript.txt"
    summary_file = "summary.txt"

    # Save transcript
    with open(transcript_file, "w", encoding="utf-8") as f:
        for entry in conversation:
            speaker = entry.get("speaker", "").capitalize()
            message = entry.get("message", "")
            f.write(f"{speaker}: {message}\n")

    # Generate summary
    generate_summary(transcript_file, summary_file)

    # Get DB connection string from env
    db_connection = os.getenv("DB_CONNECTION")
    if not db_connection:
        raise RuntimeError("❌ DB_CONNECTION not found in environment")

    # Build retriever + QA chain (LLM reused)
    retriever = build_retriever(summary_file, db_connection)
    llm = ChatOpenAI(model="o4-mini", temperature=1)
    
    template = """
    You are an assistant that must ONLY answer from the provided context.

    Context:
    {context}

    Question:
    {question}

    Answer:
    - If the answer is clearly in the context, return it.
    - If it is not, return exactly: "No information found."
    """
    prompt = PromptTemplate.from_template(template)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )

    # Process questions
    questions = updated_json.get("summary", {}).get("questions", [])
    logging.info(f"Processing {len(questions)} questions")
    for content in questions:
        if "fields" in content:
            process_fields(content["fields"], qa_chain)

    logging.info("✅ Finished processing JSON")
    return updated_json
