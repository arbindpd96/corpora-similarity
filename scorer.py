#!/usr/bin/env python3
import json
import re

import pandas as pd
import streamlit as st

from openai import OpenAI
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from db import init_db, query
from config import prompt_extract

client = OpenAI()
analyzer = SentimentIntensityAnalyzer()


@st.cache_resource
def get_db():
    return init_db()

def getSentiment(str):
    vs = analyzer.polarity_scores(str)
    return [vs['neg'], vs['neu'], vs['pos']]

def getFinalAnalysisReport(plot1, plot2, semanticAnalysisPercentage, sentimental1,
                           sentimental2):
    system_prompt = (
        "As a member of a screenwriter jury, evaluate two movie synopses based on their semantic and sentimental analysis results. Provide insights into their similarities and differences, categorized and explained clearly."
        "Task: Compare and contrast two movie synopses."
        "Respond strictly in below format"
        "Similarities between the two plots:"
        "<Similarity Category name>: Explanation of the similarity."
        "<Similarity Category name>: Explanation of the similarity."
        "..."
        "(Continue listing all similarities with their respective categories and explanations.)"
        "Differences between the two plots:"
        "\n"
        "<Difference Category name>: Explanation of the difference."
        "<Difference Category name>: Explanation of the difference."
        "..."
        "(Continue listing all differences with their respective categories and explanations.)"
        "if overall percentage is below or slightly above 50 then focus more on Differences then on Similarity. Lesser the number focus more on difference and more the number focus more on difference"
        "keep the difference strictly on the basis of story and not on basis of character's specification"
        "Don't use sentiment analysis and semantic analysis result as a comparison point use it as a context to plot differences and similarity"
    )

    system_prompt_2 = (
        "Assume you are part of a jury analyzing two movie synopses. Your task is to identify and articulate the similarities and differences in their stories, informed by semantic and sentimental analysis. Follow this structure for your response:"
        "Similarities: Identify key thematic or narrative similarities between the two synopses.\n"
        "[Similarity Category]: Provide an explanation for each similarity identified."
        "Continue listing similarity categories and explanations."
        "Differences: Highlight the main differences in story elements.\n"
        "[Difference Category]: Elaborate on each difference, focusing on story aspects rather than character specifics."
        "Ensure more emphasis on differences, especially if the overall similarity percentage is less than or slightly more than 50%."
        "Continue listing differences with their categories and explanations."
        "Note: Use the results of sentiment and semantic analysis as background context, not as primary comparison criteria and use the format mentioned above Strictly"
    )

    user_prompt = (
        "Plot1:\n"
        f"{plot1}"
        f"Sentiment analysis plot1 : {sentimental1}"
        "\n"
        "Plot2:\n"
        f"{plot2}"
        f"Sentiment analysis plot2 : {sentimental2}"
        "\n"
        f"overall similarity percentage  : {semanticAnalysisPercentage}%"
    )

    return infer_gpt_4(system_prompt=system_prompt_2, user_prompt=user_prompt)

def infer_gpt_4(system_prompt, user_prompt):
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt,
                }
            ],
            model="gpt-4-1106-preview",
            temperature=0.7
        )

        return chat_completion.choices[0].message.content
    except Exception as e:
        return e


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

def scaleTheOutput(calculatedPercentage: float):
    scaledOutput = (calculatedPercentage - 50) / 0.5
    if scaledOutput < 0:
        return 0
    return scaledOutput

def extract_categories_and_convert_to_json(response, title):
    print(f"gpt response {title}")
    print(response)
    # Define regex patterns for extracting categories and descriptions
    similarity_pattern = r'\[([^\]]+)\]:\s*([\s\S]+?)(?=\n\[|$)'
    differences_pattern = r'\[([^\]]+)\]:\s*([\s\S]+?)(?=\n\[|$)'
    overall_pattern = r'\[Overall\s*([^\]]*)\]:\s*([\s\S]+)'

    # Find all matches
    similarities = re.findall(similarity_pattern, response.split("Differences:")[0])
    differences = re.findall(differences_pattern, response.split("Differences:")[1])
    overall_match = re.search(overall_pattern, response)

    # Extract the overall overview, if available
    overall_overview = overall_match.group(2) if overall_match else ""

    # Convert the results to the desired JSON format
    result = {
        "Similarity": [{"category_name": name, "category_description": desc} for name, desc in similarities],
        "Differences": [{"category_name": name, "category_description": desc} for name, desc in differences],
        "Overall_overview": overall_overview
    }

    return json.dumps(result, indent=2)
def print_plot_comparisons(data_str):
    # Check for both 'Similarity' and 'Differences' sections
    data = json.loads(data_str)
    for section in ['Similarity', 'Differences']:
        st.subheader(f"\n{section}")
        for category in data.get(section, []):
            category_name = category.get('category_name', 'No Name')
            category_description = category.get('category_description', 'No Description')
            st.write(f"\n **{category_name}** : {category_description}")


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
        # features = extract_movie_features(prompt)
        one_liner_plot = infer_gpt_4(
            system_prompt="Based on the detailed plot provided, synthesize a one-sentence summary that encapsulates all key events and the overall story arc. Focus on merging the main plot twists, character dynamics, and the final outcome into a cohesive and succinct narrative line. Output short be short and less then 20-30 words explain entire movie motive and flow of event.",
            user_prompt=prompt)

        st.subheader("One line plot overview")

        st.markdown(one_liner_plot)

        result_df = query(collection, df, one_liner_plot)

        if 'Result' in result_df.columns and result_df['Result'].iloc[0] == "No similar plot found":
            st.write("**No similar plot found**")
        else:
            st.subheader("Showing result with >70% similarity")
            st.write(
                result_df.to_html(
                    formatters={'score': lambda x: f"{((1-x)*100):.2f}%"}
                ),
                unsafe_allow_html=True
            )
            for index, row in result_df.iterrows():
                # Extract plot, score, and title for each result
                plot = row['plot'][0] if row['plot'] else "No plot available"
                score = row['score']
                title = row['title']

                # Generate the report for each result
                report = getFinalAnalysisReport(
                    plot1=one_liner_plot,
                    plot2=plot,
                    semanticAnalysisPercentage=((1 - score) * 100),
                    sentimental1=getSentiment(prompt),
                    sentimental2=getSentiment(plot)
                )

                # Print or display the report as needed
                st.subheader(f"{title} Analysis report with similarity percentage of {round(((1 - score) * 100), 2)}%",divider=True)
                json_data = extract_categories_and_convert_to_json(report, title)
                print_plot_comparisons(json_data)



