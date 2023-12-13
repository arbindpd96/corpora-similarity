#!/usr/bin/env python3

import streamlit as st

from openai import OpenAI

from db import init_db, query
from config import prompt_extract


@st.cache_resource
def get_db():
    return init_db()


def extract_movie_features(synopsis: str) -> str:
    print("Extracting movie features...")

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
                "content": prompt_extract.format('untitled', synopsis)
            }
        ],
        max_tokens=1024
    )
    return response.choices[0].message.content


if __name__ == "__main__":

    st.set_page_config(
        page_title="Similarity Scorer",
        page_icon=":robot:"
    )

    # Initialize the database
    collection, df = get_db()

    st.title("Similarity Scorer")

    prompt = st.text_area("Synopsis", "", key="input")

    if st.button("Submit", key="inputSubmit"):        
        features = extract_movie_features(prompt)
        result_df = query(collection, df, features)

        st.write(
            result_df.to_html(
                formatters={'score': lambda x: f"{(1-x):.2f}"}
            ),
            unsafe_allow_html=True
        )
