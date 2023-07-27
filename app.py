import streamlit as st

import openai

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI

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

question = st.text_input("Input a question", "Tell me a joke")
ref_from_internet = call_langchain(question)

engineered_prompt = f"""
    Based on the context: {ref_from_internet}, 
    answer the user question: {question}
"""

answer = call_chatgpt(engineered_prompt)
st.write(answer)