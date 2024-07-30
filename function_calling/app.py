import os

import openai
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_openai import ChatOpenAI

from functions import get_exchange_rate, search_wikipedia

openai.api_key = os.environ["OPENAI_API_KEY"]

functions = [
    convert_to_openai_function(f) for f in [get_exchange_rate, search_wikipedia]
]
tools = [search_wikipedia, get_exchange_rate]


model = ChatOpenAI(temperature=0).bind(functions=functions)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Jesteś pomocnym asystantem AI, który odpowiada na pytania na podstawie wyników z wikipedii oraz podaje najnowsze kursy walut NBP",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

agent_chain = (
    RunnablePassthrough.assign(
        agent_scratchpad=lambda x: format_to_openai_functions(x["intermediate_steps"])
    )
    | prompt
    | model
    | OpenAIFunctionsAgentOutputParser()
)


memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")
agent_executor = AgentExecutor(
    agent=agent_chain, tools=tools, verbose=False, memory=memory
)


def start_app():
    while True:
        question = input("You: ")
        if question == "done":
            return

        response = agent_executor.invoke({"input": question})
        print(response["output"])


if __name__ == "__main__":
    start_app()
