import os

import numpy as np
import torch
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import RobertaTokenizer, RobertaModel

from config import EMBEDDINGS_MODEL, EMBEDDINGS_BATCH_SIZE


def getEmbeddingsUsingSentenceTransformers(inputStr: str):
    embeddings = []
    model = SentenceTransformer('gtr-t5-xl')
    for batch_start in range(0, len(inputStr), EMBEDDINGS_BATCH_SIZE):
        batch_end = batch_start + EMBEDDINGS_BATCH_SIZE
        batch = inputStr[batch_start:batch_end]
        query_embedding = model.encode(batch)
        embeddings.extend([e for e in query_embedding])

    return embeddings



