import os
import json
import mimetypes
import fitz
import docx
import pandas as pd
import re
import numpy as np
import faiss
import torch
from langchain.embeddings import HuggingFaceEmbeddings
from flask import request
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from db import get_db_connection

# === Variabel global ===
index = None
answers = []

# === Path file index & jawaban ===
INDEX_PATH = "faiss_index_dynamic.bin"

# === Load embedding model ===
embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

# === Load LLM ===
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, llm_int8_enable_fp32_cpu_offload=True)
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    return_full_text=True,
    eos_token_id=tokenizer.eos_token_id,
)

# === Ekstraksi QA dari berbagai format ===
def extract_text_from_file(file_path):
    mime_type, _ = mimetypes.guess_type(file_path)

    def extract_qa_from_text(raw_text):
        pattern = r"(?i)question\s*:\s*(.*?)\s*answer\s*:\s*(.*?)(?=\s*question\s*:|\Z)"
        matches = re.findall(pattern, raw_text, re.DOTALL | re.IGNORECASE)
        qa_pairs = []
        for q, a in matches:
            q, a = q.strip(), a.strip()
            if len(q) > 5 and len(a) > 5:
                qa_pairs.append({"question": q, "answer": a})
        return qa_pairs

    try:
        if mime_type == "application/pdf":
            doc = fitz.open(file_path)
            text = "\n".join([page.get_text("text") for page in doc])
            return extract_qa_from_text(text)

        elif mime_type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"]:
            doc = docx.Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
            return extract_qa_from_text(text)

        elif mime_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            df = pd.read_excel(file_path)
            df.columns = df.columns.str.strip().str.lower()
            if "question" in df.columns and "answer" in df.columns:
                return df[["question", "answer"]].dropna().to_dict(orient="records")
            else:
                return []

        elif mime_type == "text/plain":
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            return extract_qa_from_text(text)

        elif mime_type == "application/json":
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return [item for item in data if isinstance(item, dict) and "question" in item and "answer" in item]

        else:
            raise ValueError(f"Tipe file tidak didukung: {mime_type}")
    
    finally:
        # Hapus file setelah ekstraksi selesai
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"🧹 File {file_path} dihapus setelah diproses.")
        except Exception as e:
            print(f"⚠️ Gagal menghapus file {file_path}: {e}")

# === Prompt builder ===
def build_prompt(context, question):
    return f"""
Anda adalah asisten resmi Penerimaan Mahasiswa Baru (PMB) Politeknik Negeri Jakarta.
Anda akan diberi konteks berupa pasangan pertanyaan dan jawaban yang berkaitan dengan informasi PMB, serta sebuah pertanyaan dari pengguna.
Jawablah pertanyaan berdasarkan konteks yang tersedia.

Informasi:
{context}

Pertanyaan: {question}

Jawaban:
"""

# === CREATE INDEX DARI DATABASE ===
def create_faiss_index():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, question FROM qa_data ORDER BY id")
    records = cursor.fetchall()
    conn.close()

    if not records:
        print("⚠️ Tidak ada data QA di database.")
        return

    ids = []
    embeddings = []

    for row in records:
        q = row["question"].strip()
        emb = embedding_model.embed_documents([q])[0]
        embeddings.append(emb)
        ids.append(row["id"])

    embeddings = np.array(embeddings).astype("float32")
    base_index = faiss.IndexFlatL2(embeddings.shape[1])
    index = faiss.IndexIDMap(base_index)
    index.add_with_ids(embeddings, np.array(ids, dtype="int64"))
    faiss.write_index(index, INDEX_PATH)

    print(f"✅ FAISS index berhasil diperbarui dari MySQL. Total: {len(ids)} QA")


# === Load FAISS index dari mysql ===
def read_faiss_index():
    global index, answers

    if not os.path.exists(INDEX_PATH):
        print("⚠️ FAISS index file tidak ditemukan. Akan dibuat saat pertama upload.")
        index = None
        answers = []
        return

    print("📦 Memuat FAISS index dari file dan jawaban dari MySQL...")

    index = faiss.read_index(INDEX_PATH)

    # Ambil jawaban dari database
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, question, answer FROM qa_data")
    results = cursor.fetchall()
    conn.close()

    # Buat map: id -> {"question": ..., "answer": ...}
    answers.clear()
    max_id = max([row["id"] for row in results], default=-1)
    answers.extend([""] * (max_id + 1))
    for row in results:
        answers[row["id"]] = {
            "question": row["question"].strip(),
            "answer": row["answer"].strip()
        }

    print(f"✅ Index dan {len(results)} QA berhasil dimuat.")


# === Ambil jawaban dari index ===
def retrieve_docs(query, top_k=3, max_context=1500):
    global index, answers

    if index is None or not answers:
        print("⚠️ Index belum dimuat.")
        return ""

    query_embedding = embedding_model.embed_query(query)
    query_embedding = np.array(query_embedding).astype("float32")[np.newaxis, :]

    distances, indices = index.search(query_embedding, top_k)
    id_list = [int(i) for i in indices[0] if i >= 0]

    retrieved_qas = []
    for idx in id_list:
        if 0 <= idx < len(answers) and isinstance(answers[idx], dict):
            q = answers[idx]["question"]
            a = answers[idx]["answer"]
            retrieved_qas.append(f"Pertanyaan: {q}\nJawaban: {a}")

    context = ""
    for pair in retrieved_qas:
        if len(context + "\n\n" + pair) > max_context:
            break
        context += "\n\n" + pair

    return context.strip()

# === Generate jawaban LLM ===
# 🔹 Cek apakah input hanya sapaan tanpa niat bertanya
def is_greeting_only(text):
    text = text.lower().strip()

    # Frasa lengkap yang dianggap sapaan biasa (tidak perlu masuk RAG)
    greeting_phrases = [
        "hai", "halo", "hi", "assalamualaikum", "p", "test", "ping",
        "hai chat", "hai bot", "halo chat", "halo bot",
        "apa kabar", "halo apa kabar", "selamat pagi", "selamat malam",
        "hai semua", "halo semua", "malam", "pagi", "siang", "sore"
    ]

    # Jika input cocok persis atau mengandung frasa sapaan + tidak ada kata penting
    if any(text.startswith(phrase) or text == phrase for phrase in greeting_phrases):
        # Tambahan proteksi jika tidak mengandung kata-kata tanya berat
        question_keywords = r"\b(apa|bagaimana|kapan|dimana|siapa|mengapa|berapa|tanya|info|informasi|biaya|jalur|daftar|pmb)\b"
        if not re.search(question_keywords, text):
            return True

    return False


def generate_response(query, top_k=3):
    # Kasus hanya sapaan saja
    if is_greeting_only(query):
        return "Halo! Ada yang bisa saya bantu seputar PMB Politeknik Negeri Jakarta? 😊"

    # Tidak ada sapaan, langsung lanjut proses RAG
    context = retrieve_docs(query, top_k=top_k)
    print("🔍 Context retrieved:\n", context)
    if not context:
        return "Informasi tidak tersedia."

    prompt = build_prompt(context, query)
    try:
        outputs = generator(prompt)
        full_output = outputs[0]["generated_text"]
        response_text = full_output[len(prompt):].strip()

        for stop_token in ["\nPertanyaan:", "Pertanyaan:", "====", "\n\nPertanyaan"]:
            if stop_token in response_text:
                response_text = response_text.split(stop_token)[0].strip()

        print("📤 Output LLM:", response_text)

        if not response_text or len(response_text.split()) < 3:
            return "Informasi tidak tersedia atau tidak relevan."

        return response_text.strip()

    except Exception as e:
        print(f"🚨 Error pada LLM: {e}")
        return "Maaf, terjadi kesalahan saat menghasilkan jawaban."

def ask_handler():
    data = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return {"error": "Pertanyaan tidak boleh kosong."}, 400

    answer = generate_response(question)  # hanya 1 nilai return
    print("🟡 Pertanyaan:", question)
    print("🟢 Jawaban:", answer)

    return {
        "question": question,
        "answer": answer
    }