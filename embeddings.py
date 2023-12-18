#!/usr/bin/env python3

import csv
import os

import pandas as pd

from openai import OpenAI

from config import EMBEDDINGS_MODEL, EMBEDDINGS_BATCH_SIZE
from tornado.options import define, options
os.environ["OPENAI_API_KEY"] = 'OPENAI_KEY'

define("movies_data", default=None, help="Movies db file", type=str)

# Embeddings
embeddings_data = {
    "ids": [],
    "titles": [],
    "overview": []
}


def getEmbeddings(stringsArr: list):
    client = OpenAI()
    embeddings = []

    print("Creating embeddings...")
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


def process_item(row: dict) -> None:
    if not row['overview']:
        print(f"Skipping {row['id']} - {row['title']}")
        return

    embeddings_data["ids"].append(row['id'])
    embeddings_data["titles"].append(row['title'])
    embeddings_data["overview"].append(row['overview'])


def generate_embeddings(db_file) -> None:
    with open(db_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            process_item(row)

    df = pd.DataFrame({
        "ids": embeddings_data["ids"],
        "titles": embeddings_data["titles"],
        "embedding": getEmbeddings(embeddings_data["overview"]),
    })

    df.to_csv("data/embeddings.csv", index=False)


if __name__ == "__main__":
    options.parse_command_line()
    generate_embeddings(options.movies_data)
    print("Done!")
