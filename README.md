# Book Embeddings
Generating book embeddings via a Word2Vec-style approach using Goodreads interactions data. Books are treated as "words" and said to be in the same "sentence" if observed on the same user's shelf.

# Usage
Run `python training/run_training.py -f {filename}` to save the resulting trained neural network's state dictionary to a given file.
Run `python run_inference.py` to load in the given model and use it to generate embeddings for a given set of books. Embeddings are then written to a pinecone index for ease of serving.

# Serving
The code in `app/` allows you to run a Streamlit application which displays the top 10 most similar books by embedding of a given input book.

# Resources
This model was trained using a `g4dn.2xlarge` EC2 instance leveraging CUDA for GPU-accelerated training and inference.