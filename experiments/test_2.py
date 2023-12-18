import csv
import os

import numpy as np
import pandas as pd
import torch
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from transformers import RobertaTokenizer, RobertaModel
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config import EMBEDDINGS_MODEL, EMBEDDINGS_BATCH_SIZE
from tornado.options import define, options
from sklearn.metrics.pairwise import cosine_similarity

os.environ["OPENAI_API_KEY"] = 'OPENAI_KEY'

analyzer = SentimentIntensityAnalyzer()
client = OpenAI()


def getEmbeddingsUsingOpenAi(stringsArr: str):
    response = client.embeddings.create(
        model=EMBEDDINGS_MODEL,
        input=stringsArr
    )
    return response.data[0].embedding

def getEmbeddingsUsingSentenceTransformers(inputStr : str) :
    model = SentenceTransformer('gtr-t5-xl')
    query_embedding = model.encode(inputStr)
    return query_embedding

def getEmbeddingUsingBert(input: str) :
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaModel.from_pretrained('roberta-base')
    encoded_input_1 = tokenizer(input, return_tensors='pt')
    with torch.no_grad():
        model_output_1 = model(**encoded_input_1)
    embeddings_1 = model_output_1.last_hidden_state.mean(dim=1)
    return embeddings_1.detach().numpy()[0]


def getSentiment(str):
    vs = analyzer.polarity_scores(str)
    return [vs['neg'], vs['neu'], vs['pos']]


def infer_gpt_4(prompt):
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="gpt-4",
            temperature=0.3
        )

        return chat_completion.choices[0].message.content
    except Exception as e:
        return e


def getAttr(plot):
    prompt2 = (
        "Act like a screenplay jury. breakdown the given story into 3parts. use the story as is to create these parts. use 3 act structure"
        f"do not make up anything. keeps words absolutely simple. Rewrite in a way that we can tell them apart from another story written in a similar way. and checked with cosine similarity of the vectors using embedding"
        "if the story is not linear rewrite in a linear manner. The plot is given below. Dont acknowledge. just respond with the 3 parts. Each part should be not more than 5 words"
        "respond in a single line. comma separated. don't number them"
        "remove actors,places, and proper nouns completely"
        # "make the sentence in such a way that it should be short and showing emotion of the movie"
        "\n"
        f"{plot}"
    )

    return infer_gpt_4(prompt2)

def scaleTheOutput(calculatedPercentage : float) :
    scaledOutput = (calculatedPercentage - 50) / 0.5
    if scaledOutput < 0 :
        return 0
    return scaledOutput


data_1 = "35 year old Vimal, a married man, falls in love with a young woman and ends up killing his wife, only to discover that it was planned by his wife all along to put him behind bars."
data_2 = "A young decoit, Shubham, is recruited in the gang. When a tough situation arises, Shubham kills a policeman, only to later discover that it was planned by the gang leader to eliminate the policeman and blame Shubham."
data_3 = "Taanshi, a young vibrant head of marketing at a company, decides to run for Prime Minister of the country. Her ambitions are met with problems from all over but she doesn't give up and ends up becoming Prime Minister of the company."
data_4 = "On his dead bead, Anubhav, a tech giant, decides to name his property to his employee, Mohan, who has planned this all along. Mohan faces legal problems but finally becomes the head of the tech company and owns the property."
data_5 = "Akshat, after loosing his father, starts seeing him in his dreams telling him to sleep calmly. But his sleep is disturbed to an extent that he ends up believing his father is alive. Later he kills himself and sleeps forever."
data_6 = "Ranbir sees his childhood friend after a long time Alia and falls in love. Ranbir couldn't find her address and went for a quest of finding her address and confessing his love. Finally he met an accident while searching for Alia and the saviour was Alia and they finally met got married."


def compare(plot1, plot2, casemessage):
    str1 = getAttr(plot1)
    str2 = getAttr(plot2)

    vectorized_1 = getEmbeddingUsingBert(plot1)
    vectorized_2 = getEmbeddingUsingBert(plot2)
    print(f"shape {np.shape(vectorized_1)}")
    print(f"shape {np.shape(vectorized_2)}")

    sentiment1 = getSentiment(str1)
    sentiment2 = getSentiment(str2)

    similarity = cosine_similarity([vectorized_1], [vectorized_2])
    similarity_percentage = similarity[0][0] * 100
    emotionSimilarity = (cosine_similarity([sentiment1], [sentiment2]))[0][0] * 100
    avgOutput = ((emotionSimilarity * 2 ) + similarity_percentage ) / 3
    print("***************")
    print(str1)
    print(str2)
    print("Test Case: ", casemessage)
    print(f"Similarity Percentage: {similarity_percentage}%")
    print(f"emotion similarity Percentage: {emotionSimilarity}%")
    print(f"averaged output percentage: {avgOutput}")
    print(f"scaled output {scaleTheOutput(avgOutput)}")


compare(data_1, data_2, "high")
compare(data_3, data_4, "medium-high")
compare(data_2, data_5, "low")
