import os
from openai import OpenAI


from config import EMBEDDINGS_MODEL, EMBEDDINGS_BATCH_SIZE

client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))


def getEmbeddingsUsingOpenAi(stringsArr: list):
    embeddings = []

    for batch_start in range(0, len(stringsArr), EMBEDDINGS_BATCH_SIZE):
        batch_end = batch_start + EMBEDDINGS_BATCH_SIZE
        batch = stringsArr[batch_start:batch_end]

        response = client.embeddings.create(
            model=EMBEDDINGS_MODEL,
            input=batch
        )
        embeddings.extend([e.embedding for e in response.data])

    return embeddings
