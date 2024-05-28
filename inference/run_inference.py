import pickle
import torch


if __name__ == "__main__":
    model = torch.load("binaries/5k_steps.pt", map_location=torch.device("cpu"))
    with open("data/book_vectors.pickle", "rb") as handle:
        book_vectors = pickle.load(handle)

    model.eval()

    for vector in book_vectors.values():
        with torch.no_grad():
            book_tensor = torch.tensor(vector.astype("float32").reshape(1, -1))
            book_embedding = model.fc1(book_tensor)
