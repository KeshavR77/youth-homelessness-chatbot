import streamlit as st

import openai
# from openai import OpenAI

# client = OpenAI(api_key = st.secrets["OPENAI_API_KEY"])

# from langchain.agents import load_tools
# from langchain.agents import initialize_agent
# from langchain.agents import AgentType
# from langchain.llms import OpenAI

import numpy as np
import pandas as pd

from scipy.spatial.distance import cosine

from typing import Dict, List, Union

from langchain.document_loaders import TextLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

openai.api_key = st.secrets["OPENAI_API_KEY"]

def merge_dataframes(dataframes):
    # Concatenate the list of dataframes
    combined_dataframe = pd.concat(dataframes, ignore_index=True)

    # Ensure that the resulting dataframe only contains the columns "context", "questions", "answers"
    combined_dataframe = combined_dataframe[['context', 'questions', 'answers']]

    return combined_dataframe

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
        model="gpt-3.5-turbo-instruct",
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

# SERPAPI_API_KEY = st.secrets["SERPAPI_API_KEY"]

# def call_langchain(prompt: str) -> str:
#     llm = OpenAI(temperature=0)
#     tools = load_tools(["serpapi", "llm-math"], llm=llm)
#     agent = initialize_agent(
#         tools,
#         llm,
#         agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#         verbose=True)
    # output = agent.run(prompt)

    # return output

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

# read all csvs
# About_YSA = pd.read_csv("YSA_CSVS/About_YSA.csv")
# Board_of_Directors = pd.read_csv("YSA_CSVS/Board_of_Directors.csv")
# Definition_Of_Homeless = pd.read_csv("YSA_CSVS/Definition_Of_Homeless.csv")
# Our_Team = pd.read_csv("YSA_CSVS/Our_Team.csv")
# Programs = pd.read_csv("YSA_CSVS/Programs.csv")
# Tiny_House_Village_Application_Process = pd.read_csv("YSA_CSVS/Tiny_House_Village_Application_Process.csv")
# Tiny_House_Village_Overview = pd.read_csv("YSA_CSVS/Tiny_House_Village_Overview.csv")
# Tiny_House_Village = pd.read_csv("YSA_CSVS/Tiny_House_Village.csv")
# Youth_Leaders_Examples = pd.read_csv("YSA_CSVS/Youth_Leaders_Examples.csv")
# Youth_Leaders_Overview = pd.read_csv("YSA_CSVS/Youth_Leaders_Overview.csv")
# YSA_Supporters_Lists = pd.read_csv("YSA_CSVS/YSA_Supporters_Lists.csv")
# YSA_Supporters_Overview = pd.read_csv("YSA_CSVS/YSA_Supporters_Overview.csv")

# df = merge_dataframes([About_YSA, Board_of_Directors, Definition_Of_Homeless, Our_Team, Programs, Tiny_House_Village_Application_Process,
# Tiny_House_Village_Overview, Tiny_House_Village, Youth_Leaders_Examples, Youth_Leaders_Overview, YSA_Supporters_Lists,
# YSA_Supporters_Overview])

doc_names = {'About_YSA': 7, 'Definition_Of_Homeless': 5, 'Board_Of_Directors': 5, 'Our_Team': 2, 'Programs': 7, 'Tiny_House_Village_Application_Process': 5, 'Tiny_House_Village_Overview': 17, 'Tiny_House_Village': 8, 'Youth_Leaders_Examples': 11, 'Youth_Leaders_Overview': 1, 'YSA_Supporters_Lists': 10, 'YSA_Supporters_Overview': 1}

file_names = []

for doc, length in doc_names.items():
    file_names = file_names + ([f'YSA_TXTS/{doc}/{doc}_{i}.txt' for i in range(length)])

# Initialize an empty list to hold all documents
all_documents = [] # this is just a copy, you don't have to use this

# Iterate over each file and load its contents
for file_name in file_names:
    loader = TextLoader(file_name)
    documents = loader.load()
    all_documents.extend(documents)

# Split the loaded documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(all_documents)

# Create the open-source embedding function
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Load the documents into Chroma
db = Chroma.from_documents(docs, embedding_function)

# st.write(answer)

st.title("Youth Homelessness Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Tell me about YSA"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    question = prompt

    docs = db.similarity_search(question)

    ref_from_db_search = docs[0].page_content

    # df_screened_by_dist_score = add_dist_score_column(
    #     df, question
    # )
    # qa_pairs = convert_to_list_of_dict(df_screened_by_dist_score)

    # ref_from_internet = call_langchain(question)

    # Based on the context: {ref_from_internet}, 
    engineered_prompt = f"""
        Based on the context: {ref_from_db_search},
        answer the user question: {question}
    """

    answer = call_chatgpt(engineered_prompt)

    response = answer
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})