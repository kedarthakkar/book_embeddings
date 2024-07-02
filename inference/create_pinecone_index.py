from pinecone import Pinecone, ServerlessSpec
from app.private_constants import PINECONE_KEY


if __name__ == "__main__":
    pc = Pinecone(api_key=PINECONE_KEY)
    pc.create_index(
        name="5k-steps",
        dimension=128,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
