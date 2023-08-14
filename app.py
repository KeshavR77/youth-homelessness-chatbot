import streamlit as st

import openai

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI

import pandas as pd

from scipy.spatial.distance import cosine

from typing import Dict, List, Union

openai.api_key = st.secrets["OPENAI_API_KEY"]

def call_chatgpt(prompt: str) -> str:
    """
    Uses the OpenAI API to generate an AI response to a prompt.

    Args:
        prompt: A string representing the prompt to send to the OpenAI API.

    Returns:
        A string representing the AI's generated response.

    """

    # Use the OpenAI API to generate a response based on the input prompt.
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.5,
        max_tokens=500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )

    # Extract the text from the first (and only) choice in the response output.
    ans = response.choices[0]["text"]

    # Return the generated AI response.
    return ans

SERPAPI_API_KEY = st.secrets["SERPAPI_API_KEY"]

def call_langchain(prompt: str) -> str:
    llm = OpenAI(temperature=0)
    tools = load_tools(["serpapi", "llm-math"], llm=llm)
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True)
    output = agent.run(prompt)

    return output

def openai_text_embedding(prompt: str) -> str:
    return openai.Embedding.create(input=prompt, model="text-embedding-ada-002")[
        "data"
    ][0]["embedding"]

def calculate_sts_openai_score(sentence1: str, sentence2: str) -> float:
    # Compute sentence embeddings
    embedding1 = openai_text_embedding(sentence1)  # Flatten the embedding array
    embedding2 = openai_text_embedding(sentence2)  # Flatten the embedding array

    # Convert to array
    embedding1 = np.asarray(embedding1)
    embedding2 = np.asarray(embedding2)

    # Calculate cosine similarity between the embeddings
    similarity_score = 1 - cosine(embedding1, embedding2)

    return similarity_score

def add_dist_score_column(
    dataframe: pd.DataFrame, sentence: str,
) -> pd.DataFrame:
    dataframe["stsopenai"] = dataframe["questions"].apply(
            lambda x: calculate_sts_openai_score(str(x), sentence)
    )
    
    sorted_dataframe = dataframe.sort_values(by="stsopenai", ascending=False)


    return sorted_dataframe.iloc[:5, :]

def convert_to_list_of_dict(df: pd.DataFrame) -> List[Dict[str, str]]:
    """
    Reads in a pandas DataFrame and produces a list of dictionaries with two keys each, 'question' and 'answer.'

    Args:
        df: A pandas DataFrame with columns named 'questions' and 'answers'.

    Returns:
        A list of dictionaries, with each dictionary containing a 'question' and 'answer' key-value pair.
    """

    # Initialize an empty list to store the dictionaries
    result = []

    # Loop through each row of the DataFrame
    for index, row in df.iterrows():
        # Create a dictionary with the current question and answer
        qa_dict_quest = {"role": "user", "content": row["questions"]}
        qa_dict_ans = {"role": "assistant", "content": row["answers"]}

        # Add the dictionary to the result list
        result.append(qa_dict_quest)
        result.append(qa_dict_ans)

    # Return the list of dictionaries
    return result

df = pd.read_csv("THV-Flyer-and-application")

question = st.text_input("Input a question", "Tell me a joke")

df_screened_by_dist_score = add_dist_score_column(
    df, question
)
qa_pairs = convert_to_list_of_dict(df_screened_by_dist_score)

ref_from_internet = call_langchain(question)

engineered_prompt = f"""
    Based on the context: {ref_from_internet}, 
    and also based on the context: {qa_pairs},
    answer the user question: {question}
"""

answer = call_chatgpt(engineered_prompt)

st.write(answer)