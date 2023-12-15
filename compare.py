#!/usr/bin/env python3

import numpy as np
import pandas as pd
import streamlit as st

from openai import OpenAI
from config import EMBEDDINGS_MODEL, EMBEDDINGS_BATCH_SIZE


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


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


if __name__ == "__main__":

    st.set_page_config(
        page_title="Similarity Scorer",
        page_icon=":robot:"
    )

    st.title("Similarity Scorer")

    text = st.text_input("Original text")
    prompt = st.text_area("Compare to text lines bellow", "", key="input")

    if st.button("Submit", key="inputSubmit"):
        embeddings = getEmbeddings(
            [text] + prompt.split('\n')
        )
        df = pd.DataFrame({
            'text': prompt.split('\n'),
            'embeddings': embeddings[1:]
        })

        df["similarity"] = df.embeddings.apply(
            lambda x: cosine_similarity(x, embeddings[0])
        )
        df = df.sort_values("similarity", ascending=False)

        st.write(
            df.to_html(columns=['text', 'similarity']),
            unsafe_allow_html=True
        )
