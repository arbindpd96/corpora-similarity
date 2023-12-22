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
            print(f"processing {i} - {row['id']}, {row['title']}")
            embeddings_data["id"].append(row['id'])
            embeddings_data["title"].append(row['title'])
            embeddings_data["synopsis"].append(row['overview'])

    # Generate embeddings
    embeddings = embedding_function(embeddings_data["synopsis"])

    # Convert embeddings to JSON strings
    embeddings_json = [
        json.dumps(embedding.tolist()) if isinstance(embedding, np.ndarray) else json.dumps(embedding) for embedding
        in embeddings]

    # Create a new dictionary for processed embeddings
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
    print(proccessed_embeddings["title"])
    print(np.shape(proccessed_embeddings["embedding"]))
    df = pd.DataFrame(proccessed_embeddings)
    df.to_csv(output_file, index=False)


def get_embedding_from_csv(file, index) :
    with open(file, 'r') as file:
        reader = csv.DictReader(file)
        for i, row in enumerate(reader):
            if i == index:
                print(f"title {row['title']}")
                print(f"{type(row['embedding'])}")
                a = row['embedding'].replace('\n', ' ').replace(' ', ',')
                a = a.replace('[,', '[').replace(',]', ']')
                a = a.replace(',,',',')
                return json.loads(a)