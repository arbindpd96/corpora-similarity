from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import csv
import os

import numpy as np
import pandas as pd
from openai import OpenAI
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config import EMBEDDINGS_MODEL, EMBEDDINGS_BATCH_SIZE
from tornado.options import define, options
from sklearn.metrics.pairwise import cosine_similarity


os.environ["OPENAI_API_KEY"] = 'OPENAI_KEY'



def scale_embedding(embedding, scale_factor):
    """ Scale the embedding vector by a given factor. """
    # Ensure the embedding is a NumPy array
    embedding_array = np.array(embedding)
    return embedding_array * scale_factor

def adjust_embedding_based_on_sentiment(embedding, sentiment_scores):
    """
    Adjust the embedding based on sentiment scores.
    """
    # Ensure embedding is a NumPy array
    embedding_array = np.array(embedding)

    # Define scale factors for each sentiment component
    neg_scale = 2 + sentiment_scores['neg']
    pos_scale = 2 + sentiment_scores['pos']
    neu_scale = 1 + sentiment_scores['neu']

    # Split the embedding into three parts
    third_len = len(embedding_array) // 3
    embedding_first_third = embedding_array[:third_len]
    embedding_second_third = embedding_array[third_len:2*third_len]
    embedding_last_third = embedding_array[2*third_len:]

    # Scale each part of the embedding
    embedding_first_third_scaled = scale_embedding(embedding_first_third, neg_scale)
    embedding_second_third_scaled = scale_embedding(embedding_second_third, neu_scale)
    embedding_last_third_scaled = scale_embedding(embedding_last_third, pos_scale)

    # Combine scaled parts
    adjusted_embedding = np.concatenate([embedding_first_third_scaled,
                                         embedding_second_third_scaled,
                                         embedding_last_third_scaled])

    # Normalize the adjusted embedding
    norm = np.linalg.norm(adjusted_embedding)
    if norm > 0:
        adjusted_embedding /= norm

    return adjusted_embedding

def get_sentiment_scores(text):
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(text)

def get_embeddings(stringsArr):
    client = OpenAI()

    print("Creating embeddings...")
    print(f"Total strings: {len(stringsArr)}")

    response = client.embeddings.create(
        model=EMBEDDINGS_MODEL,
        input=stringsArr
    )

    return response.data[0].embedding

def calculateSimiliarity(vec_a, vec_b) :
    # Calculate similarity
    similarity = cosine_similarity([vec_a], [vec_b])
    print(similarity)
    similarity_percentage = similarity[0][0] * 100

    print(f"Similarity Percentage: {similarity_percentage}%")

# Example usage
data_a = "A man returns to his homeland to investigate his father's disappearance, only to discover a betrayal within his family and a call for revenge from a pro-separatist group, leading him on a path of inner turmoil and tragic love."
data_b = "A prince, guided by his father's ghost, seeks revenge on his uncle for his father's murder, leading to multiple plots, feigned madness, and a final duel that results in the deaths of all involved."
data_c = "At 14, best friends Robb Reiner and Lips made a pact to rock together forever. Their band, Anvil, hailed as the ""demi-gods of Canadian metal, "" influenced a musical generation that includes Metallica, Slayer, and Anthrax,. Following a calamitous European tour, Lips and Robb, now in their fifties, set off to record their 13th album in one last attempt to fulfill their boyhood dreams"
data_d = "At the height of the First World War, two young British soldiers must cross enemy territory and deliver a message that will stop a deadly attack on hundreds of soldiers."
data_e = "Super-assassin John Wick returns with a $14 million price tag on his head and an army of bounty-hunting killers on his trail. After killing a member of the shadowy international assassin’s guild, the High Table, John Wick is excommunicado, but the world’s most ruthless hit men and women await his every turn"

data_1 = "35 year old Vimal, a married man, falls in love with a young woman and ends up killing his wife, only to discover that it was planned by his wife all along to put him behind bars."
data_2 = "A young decoit, Shubham, is recruited in the gang. When a tough situation arises, Shubham kills a policeman, only to later discover that it was planned by the gang leader to eliminate the policeman and blame Shubham."
data_3 = "Taanshi, a young vibrant head of marketing at a company, decides to run for Prime Minister of the country. Her ambitions are met with problems from all over but she doesn't give up and ends up becoming Prime Minister of the company."
data_4 = "On his dead bead, Anubhav, a tech giant, decides to name his property to his employee, Mohan, who has planned this all along. Mohan faces legal problems but finally becomes the head of the tech company and owns the property."
data_5 = "Akshat, after loosing his father, starts seeing him in his dreams telling him to sleep calmly. But his sleep is disturbed to an extent that he ends up believing his father is alive. Later he kills himself and sleeps forever."


vectorized_a = get_embeddings(data_2)
vectorized_c = get_embeddings(data_5)
print(f"a - shape {np.shape(vectorized_a)}")
print(f"b - shape {np.shape(vectorized_c)}")

calculateSimiliarity(vectorized_a, vectorized_c)

sentiment_a = get_sentiment_scores(data_2)
sentiment_c = get_sentiment_scores(data_5)
print(f"sentiment_a {sentiment_a}")
print(f"sentiment_b {sentiment_c}")

# Adjust the embeddings based on sentiment
adjusted_vector_a = adjust_embedding_based_on_sentiment(vectorized_a, sentiment_a)
adjusted_vector_c = adjust_embedding_based_on_sentiment(vectorized_c, sentiment_c)

print(f"Shape {np.shape(adjusted_vector_a)}")
print(f"Shape {np.shape(adjusted_vector_c)}")
calculateSimiliarity(adjusted_vector_a, adjusted_vector_c)




