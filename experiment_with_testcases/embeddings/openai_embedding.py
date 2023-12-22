import os
from openai import OpenAI


from config import EMBEDDINGS_MODEL, EMBEDDINGS_BATCH_SIZE

client = OpenAI(api_key='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')


def getEmbeddingsUsingOpenAi(stringsArr: list):
    embeddings = []

    print("Creating embeddings...")
    print(f"Total strings: {len(stringsArr)}")

    for batch_start in range(0, len(stringsArr), EMBEDDINGS_BATCH_SIZE):
        batch_end = batch_start + EMBEDDINGS_BATCH_SIZE
        batch = stringsArr[batch_start:batch_end]

        print(f"\tBatch {batch_start} to {batch_end - 1}")

        response = client.embeddings.create(
            model=EMBEDDINGS_MODEL,
            input=batch
        )
        embeddings.extend([e.embedding for e in response.data])

    return embeddings
