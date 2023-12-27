#!/usr/bin/env python3

import csv
import pandas as pd

from openai import OpenAI

from config import EMBEDDINGS_MODEL, EMBEDDINGS_BATCH_SIZE, prompt_extract
from tornado.options import define, options

define("movies_data", default=None, help="Movies db file", type=str)


# Embeddings
embeddings_data = {
    "id": [],
    "title": [],
    "synopsis": [],
    # "movie_features": []
}


def getEmbeddings(stringsArr: list):
    client = OpenAI()
    embeddings = []

    print("Creating embeddings_methods...")
    print(f"Total strings: {len(stringsArr)}")

    for batch_start in range(0, len(stringsArr), EMBEDDINGS_BATCH_SIZE):
        batch_end = batch_start + EMBEDDINGS_BATCH_SIZE
        batch = stringsArr[batch_start:batch_end]

        print(f"\tBatch {batch_start} to {batch_end-1}")

        response = client.embeddings.create(
            model=EMBEDDINGS_MODEL,
            input=batch
        )

        embeddings.extend([e.embedding for e in response.data])

    return embeddings


def extract_movie_features(title: str, synopsis: str) -> str:
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful movie connoisseur."
            },
            {
                "role": "user",
                "content": prompt_extract.format(title, synopsis)
            },
        ],
        max_tokens=1024
    )
    return response.choices[0].message.content


def process_item(row: dict) -> None:
    if not row['overview']:
        print(f"Skipping {row['id']} - {row['title']}")
        return

    print(f"Processing {row['id']} - {row['title']}")

    # movie_features = extract_movie_features(
    #     row['title'],
    #     row['overview']
    # )

    # print(f"\tMovie features: {movie_features}")

    embeddings_data["id"].append(row['id'])
    embeddings_data["title"].append(row['title'])
    embeddings_data["synopsis"].append(row['overview'])
    # embeddings_data["movie_features"].append(movie_features)


def generate_embeddings(db_file) -> None:
    with open(db_file, 'r') as file:
        reader = csv.DictReader(file)
        for i, row in enumerate(reader):
            print(f"processing {i} - {row['id']}, {row['title']}")
            process_item(row)

    proccessed_embeddings = {
        "id": embeddings_data["id"],
        "title": embeddings_data["title"],
        "embedding": getEmbeddings(embeddings_data["synopsis"]),
        # "movie_features": getEmbeddings(embeddings_data["movie_features"])
    }

    df = pd.DataFrame(proccessed_embeddings)
    df.to_csv("./data/embeddings_methods.csv", index=False)


if __name__ == "__main__":
    options.parse_command_line()
    generate_embeddings('data/file.csv')
    print("Done!")
