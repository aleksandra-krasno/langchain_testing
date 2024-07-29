from glob import glob

import yaml
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings


def create_chroma_db():

    # Load configuration from YAML file
    with open("./config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Initialize embeddings
    embedding = OllamaEmbeddings(
        model=config["LLM"],
    )

    # Load all files in the DOCUMENTS_DIR directory
    files = glob(config["DOCUMENTS_DIR"] + "/*")

    loaders = [PyPDFLoader(file) for file in files]
    docs = []
    for loader in loaders:
        docs.extend(loader.load())

    print("Loaded {} files".format(len(files)))

    # Split documents into chunks of 1500 characters with overlap of 150
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    splits = text_splitter.split_documents(docs)

    print("Documents splitted into {} parts".format(len(splits)))

    # Create Chroma vector store with the documents and embeddings
    vectordb = Chroma.from_documents(
        documents=splits, embedding=embedding, persist_directory=config["DB_DIR"]
    )

    print("Save embeddings in directory {}".format(config["DB_DIR"]))


if __name__ == "__main__":
    create_chroma_db()
