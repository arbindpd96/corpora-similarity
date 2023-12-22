import numpy as np
import pytest
from sklearn.metrics.pairwise import cosine_similarity

from experiment_with_testcases.embeddings.bert_embbeding import getEmbeddingFromBert
from experiment_with_testcases.embeddings.sbert_transformer_embedding import getEmbeddingsUsingSentenceTransformers
from utils.utilities import generate_embeddings, get_embedding_from_csv


def test_sbert_embedding():
    print("testing with sbert embedding")
    generate_embeddings(embedding_function= getEmbeddingFromBert , db_file='../../data/test/sample_input.csv', output_file= '../../data/test/embeddings_bert.csv')
    print("generate embedding done")
    embedding_1 = get_embedding_from_csv('../../data/test/embeddings_bert.csv', 0)
    embedding_2 = get_embedding_from_csv('../../data/test/embeddings_bert.csv', 1)
    print(np.shape(embedding_1))
    print(type(embedding_2))

    similarity = cosine_similarity([embedding_1], [embedding_2])
    similarity_percentage = similarity[0][0] * 100
    assert similarity_percentage > 50

