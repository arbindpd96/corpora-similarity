import os
import pandas as pd
import chromadb

from ast import literal_eval
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from config import EMBEDDINGS_MODEL

os.environ["OPENAI_API_KEY"] = 'OPENAI_KEY'

# Global variables
embedding_function = OpenAIEmbeddingFunction(
    api_key=os.environ.get('OPENAI_API_KEY'),
    model_name=EMBEDDINGS_MODEL
)


def init_db():

    print("Loading data...")
    movies_df = pd.read_csv("data/embeddings.csv")
    movies_df["embedding"] = movies_df["embedding"].apply(literal_eval)
    movies_df['ids'] = movies_df['ids'].apply(str)
    movies_df['titles'] = movies_df['titles'].apply(str)

    # Create a client
    client = chromadb.EphemeralClient()

    # Create a collection
    movies_collection = client.create_collection(
        name="movies",
        embedding_function=embedding_function
    )

    print("Initializing database...")

    ids = movies_df.ids.tolist()
    embeddings = movies_df.embedding.tolist()

    # Add data to the database in baches. 
    # There is an undocumented limit in the API
    for batch_start in range(0, len(ids), 5000):
        batch_end = batch_start + 5000

        movies_collection.add(
            ids=ids[batch_start:batch_end],
            embeddings=embeddings[batch_start:batch_end]
        )

    return movies_collection, movies_df


def query(collection, movies_df, query, max_results=10) -> pd.DataFrame:
    results = collection.query(
        query_texts=query,
        n_results=max_results,
        include=['distances']
    )

    return pd.DataFrame({
        'score': results['distances'][0],
        'title': results['ids'][0]
    })
