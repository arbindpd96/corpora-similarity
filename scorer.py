#!/usr/bin/env python3
import os

import streamlit as st
from db import init_db, query

os.environ["OPENAI_API_KEY"] = 'OPENAI_KEY'

@st.cache_resource
def get_db():
    return init_db()


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
        result_df = query(collection, df, prompt)
        print(f"{result_df}")
        st.write(
            result_df.to_html(
                formatters={'score': lambda x: f"{(1-x):.2f}"}
            ),
            unsafe_allow_html=True
        )
