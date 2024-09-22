import streamlit as st
from pinecone import Pinecone
import torch
import requests
import numpy as np
import json
from private_constants import PINECONE_KEY
from book_embedding_net import BookEmbeddingNet


# Set up the model and pinecone
st.title("Book Search")

query = st.text_input("Search for a book:")
model = BookEmbeddingNet(519732, 121, 128)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state_dict = torch.load("binaries/one_epoch_state_dict.pt", map_location=device)
model.load_state_dict(state_dict)
model.eval()
pc = Pinecone(api_key=PINECONE_KEY)
index = pc.Index("one-epoch")
select_list = []

# Reads in the query if it's changed and generates the list of matching book titles
if (
    query
    and st.session_state.get("curr_query") is None
    or st.session_state.get("curr_query") != query
):
    st.session_state["curr_query"] = query
    response = requests.get(f"http://localhost:5000/search?q={query}")
    books = response.json()

    for book in books:
        select_list.append(book[0])

    st.session_state["select_list"] = select_list

# Finds the 10 most similar books to the chosen book title
selected_option = st.selectbox(
    "Select an option", st.session_state.get("select_list", [])
)

if selected_option:
    # Get the vector from the database
    book_info = requests.get(f"http://localhost:5000/book?q={selected_option}")
    vector = book_info.json()[0]

    with torch.no_grad():
        book_tensor = torch.tensor(np.array(vector).astype("float32").reshape(1, -1))
        book_embedding = model.fc1(book_tensor)

    query_results = index.query(
        namespace="book-vectors",
        vector=book_embedding.numpy().flatten().tolist(),
        top_k=10,
        include_values=True,
    )

    results = query_results.to_dict()["matches"]
    embedding_id_list = [result["id"][9:] for result in results]
    similar_books = requests.get(
        f"http://localhost:5000/books?values={json.dumps(embedding_id_list)}"
    )
    similar_book_list = []

    for book in json.loads(similar_books.content):
        similar_book_list.append(book[0])

    st.write(similar_book_list)
