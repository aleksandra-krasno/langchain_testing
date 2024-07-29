import os

import yaml
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma


def start_app():

    # Load configuration from YAML file
    with open("./config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if not os.path.exists(config["DB_DIR"]):
        print("Chroma db does not exist! Run `python db.py` first. ")
        return

    embedding = OllamaEmbeddings(
        model=config["LLM"],
    )

    # Load db and model
    vectordb = Chroma(persist_directory=config["LLM"], embedding_function=embedding)
    model = ChatOllama(model=config["LLM"])

    # Run chain
    QA_CHAIN_PROMPT = PromptTemplate.from_template(config["PROMPT"])

    qa_chain = RetrievalQA.from_chain_type(
        model,
        retriever=vectordb.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    )

    while True:
        question = input("You: ")
        if question == "stop":
            return

        answer = qa_chain.invoke({"query": question})
        print(answer["result"])


if __name__ == "__main__":
    start_app()
