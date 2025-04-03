import os
import mysql.connector
from flask import Flask, request, jsonify
import openai
from pinecone import Pinecone, ServerlessSpec
import json
from datetime import datetime
from flask_cors import CORS
from dotenv import load_dotenv
load_dotenv()

# Initialize Flask App
app = Flask(__name__)
CORS(app)

# Hardcoded API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX_NAME = "applicationjson"

# Initialize OpenAI
openai.api_key = OPENAI_API_KEY

# Initialize Pinecone Client
pc = Pinecone(api_key=PINECONE_API_KEY)
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=1024,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
pinecone_index = pc.Index(PINECONE_INDEX_NAME)

# Database Connection
def get_db_connection():
    try:
        db_conn = mysql.connector.connect(
            host=os.getenv("DB_HOST"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_NAME")
        )
        return db_conn
    except mysql.connector.Error as e:
        print(f"Error connecting to MySQL database: {e}")
        return None

# Serialize Pinecone Results

def serialize_pinecone_results(results):
    return [
        {"id": result.id, "score": result.score, "metadata": result.metadata}
        for result in results
    ]

# Get Last 5 Logs for Chain of Thought

def get_last_five_logs(application_id):
    db_conn = get_db_connection()
    if db_conn is None:
        return []

    try:
        cursor = db_conn.cursor(dictionary=True)
        query = """
            SELECT query, response 
            FROM log_1 
            WHERE application_id = %s 
            ORDER BY id DESC 
            LIMIT 1;
        """
        cursor.execute(query, (application_id,))
        logs = cursor.fetchall()
        cursor.close()
        return logs[::-1]
    except mysql.connector.Error as e:
        print(f"Database error fetching logs: {e}")
        return []
    finally:
        db_conn.close()

# Vectorize Query

def get_query_vector(query):
    try:
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=query
        )
        return response['data'][0]['embedding']
    except Exception as e:
        print(f"Error generating query vector: {e}")
        return None

# Semantic Search

def query_pinecone_semantic(query_vector):
    if query_vector is None:
        return []

    try:
        results = pinecone_index.query(
            vector=query_vector,
            top_k=50,
            include_metadata=True
        )
        return results.get("matches", [])
    except Exception as e:
        print(f"Error querying Pinecone: {e}")
        return []

# Construct Context

def construct_context(pinecone_results):
    if not pinecone_results:
        return "No relevant data available in context."

    context_lines = []
    for result in pinecone_results:
        metadata = result.get("metadata", {})
        for key, value in metadata.items():
            context_lines.append(f"{key.replace('_', ' ').title()}: {value}")
    return "\n".join(context_lines)

# Generate Answer

def answer_query_with_openai(query, context, history_context=""):
    try:
        prompt = f"""
You are a senior credit underwriter assessing a business loan application in the Indian regulatory and financial context. Use the chain of thought technique by analyzing current and past queries to deepen your understanding. Analyze the application using only explicitly provided data—no assumptions or hypotheticals.

---

Conversation History:
{history_context if history_context else "No prior queries."}

---

Relevant Context:
{context}

---

New Query:
{query}

Answer:
"""
        response = openai.ChatCompletion.create(
            model="gpt-4.5-preview",
            messages=[
                {"role": "system", "content": "You are a financial assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Error generating response with OpenAI: {e}")
        return "An error occurred while generating the response."

# Query Handler

@app.route('/query', methods=['POST'])
def handle_query():
    try:
        data = request.json
        query = data.get("query")
        application_id = data.get("application_id")

        if not query:
            return jsonify({"error": "Query is required"}), 400
        if not application_id:
            return jsonify({"error": "Application ID is required"}), 400

        query_vector = get_query_vector(query)
        if not query_vector:
            return jsonify({"error": "Failed to generate query vector"}), 500

        pinecone_results = query_pinecone_semantic(query_vector)
        if not pinecone_results:
            return jsonify({"error": "No relevant data found"}), 404

        context = construct_context(pinecone_results)

        # Fetch last 5 queries for chain of thought
        conversation_history = get_last_five_logs(application_id)
        history_lines = [f"Q{idx+1}: {log_1['query']}\nA{idx+1}: {log_1['response']}" for idx, log_1 in enumerate(conversation_history)]
        history_context = "\n\n".join(history_lines)

        response = answer_query_with_openai(query, context, history_context)

        db_conn = get_db_connection()
        cursor = db_conn.cursor()
        metadata_json = json.dumps({"matched_files": serialize_pinecone_results(pinecone_results)}) if pinecone_results else None
        query_str = """
            INSERT INTO log_1 (query, application_id, response, metadata)
            VALUES (%s, %s, %s, %s);
        """
        cursor.execute(query_str, (query, application_id, response, metadata_json))
        db_conn.commit()
        query_id = cursor.lastrowid
        cursor.close()
        db_conn.close()

        return jsonify({"query": query, "response": response, "query_id": query_id})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Internal error: {str(e)}"}), 500

# Save Correction

def update_correction_in_db(query_id, correction):
    db_conn = get_db_connection()
    if db_conn is None:
        return {"error": "Database connection unavailable"}

    try:
        cursor = db_conn.cursor()
        update_query = """
            UPDATE log_1 
            SET correction = %s
            WHERE id = %s;
        """
        cursor.execute(update_query, (correction, query_id))
        db_conn.commit()
        cursor.close()
        return {"message": "Correction updated successfully"}
    except mysql.connector.Error as e:
        return {"error": str(e)}
    finally:
        db_conn.close()

@app.route("/save_correction", methods=["POST"])
def save_correction():
    data = request.get_json()
    correction_text = data.get("correction", "").strip()
    query_id = data.get("query_id")

    if not correction_text:
        return jsonify({"error": "Correction text is required"}), 400
    if not query_id:
        return jsonify({"error": "Query ID is required"}), 400

    try:
        result = update_correction_in_db(query_id, correction_text)
        if "error" in result:
            return jsonify(result), 500
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run Flask App
if __name__ == "__main__":
    try:
        app.run(debug=True, port=5000)
    except Exception as e:
        print(f"Error starting Flask server: {e}")
