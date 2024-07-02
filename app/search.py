from flask import Flask, request, jsonify
import psycopg2
import json

app = Flask(__name__)
DB_NAME = "books_v3"


def get_db_connection():
    conn = psycopg2.connect(dbname="postgres", user="kedarthakkar", host="localhost")
    return conn


@app.route("/search", methods=["GET"])
def search_books():
    query = request.args.get("q", "")
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        f"SELECT book_title FROM {DB_NAME} WHERE book_title ILIKE %s LIMIT 50",
        (f"%{query}%",),
    )
    books = cur.fetchall()
    cur.close()
    conn.close()
    return jsonify(books)


@app.route("/book", methods=["GET"])
def book_info():
    query = request.args.get("q", "")
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        f"SELECT book_vector FROM {DB_NAME} WHERE book_title ILIKE %s LIMIT 1",
        (f"%{query}%",),
    )
    books = cur.fetchall()
    cur.close()
    conn.close()
    return jsonify(books)


@app.route("/books", methods=["GET"])
def similar_books():
    books = request.args.getlist("values")
    conn = get_db_connection()
    cur = conn.cursor()
    book_parse = ",".join(json.loads(books[0]))
    cur.execute(f"SELECT book_title FROM books_v3 WHERE embedding_id IN ({book_parse})")
    books = cur.fetchall()
    cur.close()
    conn.close()
    return jsonify(books)


if __name__ == "__main__":
    app.run(debug=True)
