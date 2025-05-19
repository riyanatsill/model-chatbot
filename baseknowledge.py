import os
from flask import request, session
from db import get_db_connection
from model import extract_text_from_file, create_faiss_index, read_faiss_index
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'data'

def upload_file_handler():
    if 'file' not in request.files:
        return {"message": "Tidak ada file yang dikirim."}, 400

    file = request.files['file']
    if file.filename == '':
        return {"message": "Nama file kosong."}, 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        qa_pairs = extract_text_from_file(filepath)
        print("✅ Jumlah QA ditemukan:", len(qa_pairs))
        if not qa_pairs:
            return {"message": "File tidak mengandung QA valid."}, 400

        uploaded_by = session.get("user", {}).get("username", "unknown")

        # Simpan file ke uploaded_files
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO uploaded_files (filename, uploaded_by)
            VALUES (%s, %s)
            ON DUPLICATE KEY UPDATE uploaded_at = NOW(), uploaded_by = %s
        """, (filename, uploaded_by, uploaded_by))

        # Simpan QA langsung ke qa_data
        for qa in qa_pairs:
            cursor.execute("""
            INSERT INTO qa_data (question, answer, filename, created_by)
            VALUES (%s, %s, %s, %s)
        """, (qa["question"], qa["answer"], filename, uploaded_by))


        conn.commit()
        conn.close()

        # Update FAISS index dari database
        create_faiss_index()
        read_faiss_index()

        return {"message": f"File {filename} berhasil diupload dan QA berhasil ditambahkan."}, 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"message": f"Gagal memproses file: {e}"}, 500
    
def list_uploaded_files_handler():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT uf.filename, uf.uploaded_by, uf.uploaded_at,
               COUNT(q.id) AS total_qa
        FROM uploaded_files uf
        LEFT JOIN qa_data q ON uf.filename = q.filename
        GROUP BY uf.filename
        ORDER BY uf.uploaded_at DESC
    """)
    files = cursor.fetchall()
    cursor.close()
    conn.close()
    return {"files": files}

def delete_file_handler(filename):
    file_path = os.path.join(UPLOAD_FOLDER, filename)

    conn = get_db_connection()
    cursor = conn.cursor()

    # Hapus dari DB
    cursor.execute("DELETE FROM uploaded_files WHERE filename = %s", (filename,))
    cursor.execute("DELETE FROM qa_data WHERE filename = %s", (filename,))
    conn.commit()
    cursor.close()
    conn.close()

    # Hapus file fisik
    if os.path.exists(file_path):
        os.remove(file_path)

    # Rebuild FAISS index dari data sisa
    create_faiss_index()

    return {"message": f"File {filename} dan seluruh QA terkait berhasil dihapus."}