# === app.py ===
from flask import Flask, jsonify
from flask_session import Session
from flask_cors import CORS
import os
from dotenv import load_dotenv
#from model import read_faiss_index
from model import read_faiss_index, ask_handler
from baseknowledge import upload_file_handler, delete_file_handler

load_dotenv()
# === SETUP ===
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)
CORS(app, supports_credentials=True, origins=["https://frontend-chatbot-beta.vercel.app/"])

UPLOAD_FOLDER = 'data'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# === Load FAISS saat startup ===
read_faiss_index()

# === USER ===
@app.route("/ask", methods=["POST"])
def ask_route():
    result = ask_handler()
    return jsonify(result)

# === BASE KNOWLEDGE ===
@app.route('/upload', methods=['POST'])
def upload_file():
    return upload_file_handler()

@app.route('/delete-file/<filename>', methods=['DELETE'])
def delete_file(filename):
    result = delete_file_handler(filename)
    return jsonify(result)


# === MAIN ===
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
