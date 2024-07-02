import pickle
import torch
from pinecone import Pinecone
from app.private_constants import PINECONE_KEY
from training.book_embedding_net import BookEmbeddingNet
from training.constants import BOOK_FEATURE_DIM, NUM_BOOKS, EMBEDDING_DIM


if __name__ == "__main__":
    """
    Generates book embeddings for every book in the vocabulary and writes them to
    Pinecone in batches of size 500.
    """
    model = BookEmbeddingNet(NUM_BOOKS, BOOK_FEATURE_DIM, EMBEDDING_DIM)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load("binaries/5k_steps_state_dict.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    pc = Pinecone(api_key=PINECONE_KEY)
    with open("data/book_vectors.pickle", "rb") as handle:
        book_vectors = pickle.load(handle)

    upsert_list = []
    index = pc.Index("5k-steps")

    for i, vector in book_vectors.items():
        with torch.no_grad():
            book_tensor = torch.tensor(vector.astype("float32").reshape(1, -1))
            book_embedding = model.fc1(book_tensor)
            upsert_list.append(
                {
                    "id": f"book_vec_{i}",
                    "values": book_embedding.numpy().flatten().tolist(),
                }
            )

        if (i + 1) % 500 == 0:
            print("Inserting into Pinecone index")
            index.upsert(
                vectors=upsert_list,
                namespace="test-ns",
            )
            upsert_list = []

    print("Inserting into Pinecone index")
    index.upsert(
        vectors=upsert_list,
        namespace="test-ns",
    )
