import os
import mysql.connector
from flask import Flask, request, jsonify
import openai
from pinecone import Pinecone
from pinecone import ServerlessSpec
import json
from datetime import datetime

# Initialize Flask App
app = Flask(__name__)

# Hardcoded API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX_NAME = "applicationjson"

# Initialize OpenAI
openai.api_key = OPENAI_API_KEY

# Initialize Pinecone Client (NEW METHOD)
pc = Pinecone(api_key=PINECONE_API_KEY)

# Check if index exists, else create it
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX_NAME, 
        dimension=1536,  # Update to match OpenAI's embedding size
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Connect to existing Pinecone index
pinecone_index = pc.Index(PINECONE_INDEX_NAME)

# Initialize MySQL Connection
try:
    db_conn = mysql.connector.connect(
        host="crossover.proxy.rlwy.net",
        port=57490,
        user="root",
        password="yKbGScLCUBxgrzZRyXIrCixYUWuQjJIE",
        database="Credable"
    )
except mysql.connector.Error as e:
    print(f"Error connecting to MySQL database: {e}")
    db_conn = None

def get_db_connection():
    """Establish and return a database connection."""
    try:
        db_conn = mysql.connector.connect(
            host=os.getenv("DB_HOST", "localhost"),
            user=os.getenv("DB_USER", "root"),
            password=os.getenv("DB_PASSWORD", "aisha@123"),
            database=os.getenv("DB_NAME", "Credable")
        )
        return db_conn
    except mysql.connector.Error as e:
        print(f"Error connecting to MySQL database: {e}")
        return None

def serialize_pinecone_results(results):
    """Convert Pinecone ScoredVector objects into JSON-serializable format."""
    serialized_results = []
    for result in results:
        serialized_results.append({
            "id": result.id,
            "score": result.score,
            "metadata": result.metadata
        })
    return serialized_results

def save_log_to_db(query, application_id, response, metadata=None):
    """Save query, response, and metadata to the log table."""
    db_conn = get_db_connection()
    if db_conn is None:
        print("Database connection is not available.")
        return

    try:
        cursor = db_conn.cursor()
        
        # ✅ Convert Pinecone results to a JSON-serializable format
        if metadata and "matched_files" in metadata:
            metadata["matched_files"] = serialize_pinecone_results(metadata["matched_files"])

        metadata_json = json.dumps(metadata) if metadata else None
        query_str = """
            INSERT INTO log (query, application_id, response, metadata)
            VALUES (%s, %s, %s, %s);
        """
        cursor.execute(query_str, (query, application_id, response, metadata_json))
        db_conn.commit()
        cursor.close()
    except mysql.connector.Error as e:
        print(f"Database error: {e}")
    finally:
        db_conn.close()


def get_query_vector(query):
    """Convert the query into a vector using OpenAI embeddings."""
    try:
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=query
        )
        return response['data'][0]['embedding']
    except Exception as e:
        print(f"Error generating query vector: {e}")
        return None


def query_pinecone_semantic(query_vector):
    """Perform semantic search in Pinecone using the query vector."""
    if query_vector is None:
        return []

    try:
        results = pinecone_index.query(
            vector=query_vector,
            top_k=50,  # Number of results to retrieve
            include_metadata=True
        )

        if "matches" in results:
            return results["matches"]
        else:
            return []
    except Exception as e:
        print(f"Error querying Pinecone: {e}")
        return []


def construct_context(pinecone_results):
    """Dynamically construct context from Pinecone query results."""
    if not pinecone_results:
        return "No relevant data available in context."

    context_lines = []
    for result in pinecone_results:
        metadata = result.get("metadata", {})
        for key, value in metadata.items():
            context_lines.append(f"{key.replace('_', ' ').title()}: {value}")
    return "\n".join(context_lines)


def answer_query_with_openai(query, context):
    """Generate a response using OpenAI GPT with the provided query and context."""
    try:
        prompt = f"""
You are a senior credit underwriter analyzing a loan application. Understand the indian underwriting scenario and the indian business registrations and terms, then analyse the loan application, background , financials, company strcuture, triangulate all the data provided across company and provide data and numbers- add rationale. Do not make assumptions or include information that is not explicitly provided.

Relevant Context:
{context}

Query:
{query}

Answer:
"""
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Use "gpt-4" for better performance if needed
            messages=[
                {"role": "system", "content": "You are a financial assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Error generating response with OpenAI: {e}")
        return "An error occurred while generating the response."


@app.route('/query', methods=['POST'])
def handle_query():
    """Handle incoming queries and generate responses."""
    try:
        data = request.json
        query = data.get("query")
        application_id = data.get("application_id")  # ✅ Extract application_id from request

        if not query:
            return jsonify({"error": "Query is required"}), 400
        
        if not application_id:
            return jsonify({"error": "Application ID is required"}), 400  # ✅ Ensure application_id is provided

        # Generate query vector
        query_vector = get_query_vector(query)
        if not query_vector:
            return jsonify({"error": "Failed to generate query vector"}), 500

        # Perform semantic search in Pinecone
        pinecone_results = query_pinecone_semantic(query_vector)
        if not pinecone_results:
            return jsonify({"error": "No relevant data found"}), 404

        # Dynamically construct context
        context = construct_context(pinecone_results)
        print(f"Constructed Context: {context}")  # Debug log

        # Generate a response using OpenAI
        response = answer_query_with_openai(query, context)
        print(f"OpenAI Response: {response}")  # Debug log

        # Save log to the database with application_id
        save_log_to_db(query, application_id, response, metadata={"matched_files": pinecone_results})

        return jsonify({"query": query, "response": response})
    
    except Exception as e:
        print(f"Error handling the query: {e}")  # Full error printout
        import traceback
        traceback.print_exc()  # Print full stack trace for debugging
        return jsonify({"error": f"Internal error: {str(e)}"}), 500


# Start the Flask server
if __name__ == "__main__":
    try:
        app.run(debug=True, port=5007)
    except Exception as e:
        print(f"Error starting Flask server: {e}")
