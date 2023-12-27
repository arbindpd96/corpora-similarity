import csv
import json

import numpy as np
import pandas as pd

embeddings_data = {
    "id": [],
    "title": [],
    "synopsis": [],
}
def generate_embeddings(embedding_function, db_file, output_file) -> None:
    with open(db_file, 'r') as file:
        reader = csv.DictReader(file)
        for i, row in enumerate(reader):
            embeddings_data["id"].append(row['id'])
            embeddings_data["title"].append(row['title'])
            embeddings_data["synopsis"].append(row['overview'])

    # Generate embeddings_methods
    embeddings = embedding_function(embeddings_data["synopsis"])

    # Convert embeddings_methods to JSON strings
    embeddings_json = [
        json.dumps(embedding.tolist()) if isinstance(embedding, np.ndarray) else json.dumps(embedding) for embedding
        in embeddings]

    # Create a new dictionary for processed embeddings_methods
    proccessed_embeddings = {
        "id": embeddings_data["id"],
        "title": embeddings_data["title"],
        "embedding": embeddings_json
    }
    # proccessed_embeddings = {
    #     "id": embeddings_data["id"],
    #     "title": embeddings_data["title"],
    #     "embedding": embedding_function(embeddings_data["synopsis"]),
    # }
    df = pd.DataFrame(proccessed_embeddings)
    df.to_csv(output_file, index=False)


def get_embedding_from_csv(file, index) :
    with open(file, 'r') as file:
        reader = csv.DictReader(file)
        for i, row in enumerate(reader):
            if i == index:
                a = row['embedding'].replace('\n', ' ').replace(' ', ',')
                a = a.replace('[,', '[').replace(',]', ']')
                a = a.replace(',,',',')
                return json.loads(a)