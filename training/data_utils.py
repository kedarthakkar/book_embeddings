import pandas as pd
from constants import DECADE_BINS, DECADE_LABELS
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import spacy
import torch
import pickle


def interaction_df_to_book_vectors(
    interaction_df: pd.DataFrame, book_to_index: dict[int, int]
) -> dict[int, list[float]]:
    """
    Maps the interaction dataframe to a collection of book vectors. These are 121-d
    numeric vectors that represent information about the book, normalized and scaled, along
    with one-hot encoding for categorical variables.
    """
    book_embedding_df = interaction_df.copy()
    book_embedding_df = book_embedding_df.drop_duplicates(subset="book_id").reset_index(
        drop=True
    )
    book_embedding_df = book_embedding_df.drop(
        ["user_id", "country_code", "format"], axis=1
    )
    book_embedding_df["is_ebook"] = book_embedding_df["is_ebook"].apply(
        lambda x: int(x)
    )
    book_embedding_df["publication_decade"] = pd.cut(
        book_embedding_df["publication_year"],
        bins=DECADE_BINS,
        labels=DECADE_LABELS,
        right=False,
    )
    book_embedding_df = book_embedding_df.drop(["publication_year"], axis=1)
    book_embedding_df = pd.get_dummies(
        book_embedding_df, columns=["publication_month", "publication_decade"]
    )

    scaler = MinMaxScaler()
    columns_to_normalize = [
        "rating",
        "text_reviews_count",
        "average_rating",
        "num_pages",
        "ratings_count",
    ]
    book_embedding_df[columns_to_normalize] = scaler.fit_transform(
        book_embedding_df[columns_to_normalize]
    )
    book_vectors = {}
    nlp = spacy.load("en_core_web_md")

    for _, row in book_embedding_df.iterrows():
        book_id = row["book_id"]
        description_embed = nlp(row["description"]).vector[:64]
        title_embed = nlp(row["title"]).vector[:32]
        feature_vec = row.drop(["description", "title", "book_id"]).values
        book_vec = np.concatenate([feature_vec, description_embed, title_embed], axis=0)
        book_vectors[book_to_index[book_id]] = book_vec

    with open("data/book_vectors.pickle", "wb") as handle:
        pickle.dump(book_vectors, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return book_vectors


def get_sparse_tensor(
    interaction_df: pd.DataFrame,
    user_to_index: dict[int, int],
    book_to_index: dict[int, int],
):
    """
    Maps book ID to sparse vector representing context books (books in the same GoodReads shelves).
    """
    def get_label_id(book_ids):
        return [book_to_index[book_id] for book_id in book_ids]

    grouped_interactions = (
        interaction_df[["user_id", "book_id"]]
        .groupby("user_id")["book_id"]
        .apply(get_label_id)
        .reset_index()
    )
    sparse_tensor = torch.zeros(
        len(user_to_index), len(book_to_index), dtype=torch.float
    )

    for _, row in grouped_interactions.iterrows():
        user_index = user_to_index[row["user_id"]]

        for book_index in row["book_id"]:
            sparse_tensor[user_index, book_index] = 1.0

    return sparse_tensor, grouped_interactions
