import csv
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics.pairwise import cosine_similarity

from experiment_with_testcases.embeddings_methods.openai_embedding import getEmbeddingsUsingOpenAi
from utils.utilities import generate_embeddings, get_embedding_from_csv


def test_openai_embedding():
    print("testing with openai embedding")
    generate_embeddings(embedding_function= getEmbeddingsUsingOpenAi , db_file='../../data/test/sample_input.csv', output_file= '../../data/test/embeddings_openai.csv')
    embedding_1 = get_embedding_from_csv('../../data/test/embeddings_openai.csv', 0)
    embedding_2 = get_embedding_from_csv('../../data/test/embeddings_openai.csv', 1)

    similarity = cosine_similarity([embedding_1], [embedding_2])
    similarity_percentage = similarity[0][0] * 100
    assert similarity_percentage > 70

