import torch.nn as nn


class BookEmbeddingNet(nn.Module):
    """
    Neural net to generate book embeddings based on GoodReads interaction data.
    The first layer maps a 121-d vector of book features to a 128-d embedding.
    The output is the likelihood of every book in the vocabulary appearing in context
    with the input.
    """
    def __init__(self, num_books, book_feature_dim, embedding_dim):
        super(BookEmbeddingNet, self).__init__()
        self.fc1 = nn.Linear(book_feature_dim, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, num_books)

    def forward(self, book_features):
        # Pass book features through fully connected layers
        hidden_layer = self.fc1(book_features)
        output_layer = self.fc2(hidden_layer)
        return output_layer
