import pandas as pd
from data_utils import interaction_df_to_book_vectors
from data_utils import get_sparse_tensor
from book_embedding_net import BookEmbeddingNet
import torch.nn as nn
import torch.optim as optim
import torch
import argparse
import os
import pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename")
    args = parser.parse_args()

    # Loads the aggregated user-book interaction data
    if os.path.exists("data/train_data.csv"):
        interaction_df = pd.read_csv("data/train_data.csv")
    else:
        raise FileNotFoundError("User-book interaction data not found")
    print("Done loading interaction DF")

    # Map user and book IDs to embedding IDs starting from 1
    user_to_index = {
        user_id: i for i, user_id in enumerate(interaction_df["user_id"].unique())
    }
    book_to_index = {
        book_id: i for i, book_id in enumerate(interaction_df["book_id"].unique())
    }

    # Aggregate map of book ID (raw) --> 121-d vector
    if os.path.exists("data/book_vectors.pickle"):
        print("Loading book vectors from file")
        with open("data/book_vectors.pickle", "rb") as handle:
            book_vectors = pickle.load(handle)
    else:
        print("Building book vectors")
        book_vectors = interaction_df_to_book_vectors(interaction_df, book_to_index)
    print("Done building book vectors")

    # Generate sparse tensor and grouped interactions
    sparse_tensor, grouped_interactions = get_sparse_tensor(
        interaction_df, user_to_index, book_to_index
    )
    print("Done generating training data")

    # Run training
    model = BookEmbeddingNet(len(book_to_index), 121, 128)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    num_epochs = 1
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, row in grouped_interactions.iterrows():
            if (i + 1) > 1:
                break

            user_index = user_to_index[row["user_id"]]
            labels = sparse_tensor[user_index]

            for book_index in row["book_id"]:
                book_features = torch.tensor(list(book_vectors[book_index]))
                outputs = model(book_features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            if (i + 1) % 1 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(grouped_interactions)}], Loss: {running_loss / 100:.4f}"
                )
                running_loss = 0.0

    torch.save(model, f"binaries/{args.filename}")
    print("Finished Training")
