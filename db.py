import csv
import sqlite3

if sqlite3.sqlite_version_info < (3, 35, 0):
    # hotswap to pysqlite-binary if it's too old
    import sys

    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import os
import pandas as pd
import chromadb

from ast import literal_eval
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from config import EMBEDDINGS_MODEL

# os.environ.setdefault('OPENAI_API_KEY', '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')
# Global variables
embedding_function = OpenAIEmbeddingFunction(
    api_key=os.environ.get('OPENAI_API_KEY'),
    model_name=EMBEDDINGS_MODEL
)

def init_db():

    print("Loading data...")
    movies_df = pd.read_csv("data/embeddings.csv")

    movies_df['id'] = movies_df['id'].apply(str)
    movies_df["embedding"] = movies_df["embedding"].apply(literal_eval)
    # movies_df["movie_features"] = movies_df["movie_features"].apply(literal_eval)

    # Create a client
    client = chromadb.EphemeralClient()

    # Create a collection
    movies_collection = client.create_collection(
        name="movies",
        embedding_function=embedding_function
    )

    print("Initializing database...")

    ids = movies_df.id.tolist()
    embeddings = movies_df.embedding.tolist()  # movies_df.embedding.tolist()
    #
    # Add data to the database in baches. 
    # There is an undocumented limit in the API
    for batch_start in range(0, len(ids), 5000):
        batch_end = batch_start + 5000

        movies_collection.add(
            ids=ids[batch_start:batch_end],
            embeddings=embeddings[batch_start:batch_end]
        )

    return movies_collection, movies_df


def query(collection, movies_df, query, max_results=3) -> pd.DataFrame:
    print("Querying database...")

    results = collection.query(
        query_texts=query,
        n_results=max_results,
        include=['distances']
    )
    print(results['ids'])
    print(results['distances'])
    print(f"{get_plot_for_selected_id('data/file.csv', results['ids'][0])}")
    distances = []
    ids = []

    for index, distance in enumerate(results['distances'][0]) :
        print(f"index {index} and {distance}")
        if ((1-distance) * 100) > 70:
            distances.append(distance)
            ids.append(results['ids'][0][index])

    print(distances)
    print(ids)
    if len(distances) == 0 :
        print("No data found")
        return pd.DataFrame({
            "Result" : "No similar plot found"
        },index= [0])
    else :
        print("data found")
        return pd.DataFrame({
            'score': distances,
            'title': movies_df[movies_df.id.isin(ids)]['title'],
            'plot' : get_plot_for_selected_id('data/file.csv', ids)
        })

def get_plot_for_selected_id(db_file, id) -> []:
    overviews = []
    with open(db_file, 'r') as file:
        reader = csv.DictReader(file)
        for i, row in enumerate(reader):
            for i in id:
                if row['id'] == str(i):
                    overviews.append(row['overview'])

    return overviews
