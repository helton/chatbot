import os

from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings import (
    CacheBackedEmbeddings,
    GPT4AllEmbeddings,
    OllamaEmbeddings,
    OpenAIEmbeddings,
)
from langchain.llms import Ollama
from langchain.chat_models import ChatOpenAI
from langchain.llms.base import BaseLLM
from langchain.prompts import PromptTemplate
from langchain.schema.embeddings import Embeddings
from langchain.storage import RedisStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, FAISS
from typing import List, Dict

from dotenv import load_dotenv

load_dotenv()


def get_providers():
    return {
        "embeddings": {
            "bert": "gpt4all",
            "llama2": "ollama",
            "mistral": "ollama",
            "text-embedding-ada-002": "openai",
        },
        "llms": {
            "gpt-3.5-turbo": "openai",
            "gpt-4": "openai",
            "mistral": "ollama",
            "llama2": "ollama",
        },
    }


def load_embedding(embedding_model: str, keys: Dict[str, str]) -> Embeddings:
    embeddings_providers_map = {
        # dim=384, model=bert
        "gpt4all": lambda: GPT4AllEmbeddings(),
        # dim=4096, model=llama2
        # dim=4096, model=mistral
        "ollama": lambda: OllamaEmbeddings(model=embedding_model),
        # dim=1536, model=text-embedding-ada-002
        "openai": lambda: OpenAIEmbeddings(
            model=embedding_model, openai_api_key=keys["OPENAI_API_KEY"]
        ),
    }
    providers = get_providers()
    return embeddings_providers_map[providers["embeddings"][embedding_model]]()


def load_cached_embedding(embedding_model: str, keys: Dict[str, str]) -> Embeddings:
    store = RedisStore(
        redis_url=f'redis://{os.environ["REDIS_HOST"]}:{os.environ["REDIS_PORT"]}',
        client_kwargs={"db": int(os.environ["REDIS_DB"])},
        namespace=os.environ["REDIS_NAMESPACE"],
    )

    underlying_embeddings = load_embedding(embedding_model, keys=keys)
    return CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings, store, namespace=embedding_model
    )


def load_llm(llm_model: str, keys: Dict[str, str]) -> BaseLLM:
    llm_provider_map = {
        "openai": lambda: ChatOpenAI(
            model_name=llm_model,
            temperature=0,
            openai_api_key=keys["OPENAI_API_KEY"],
        ),
        "ollama": lambda: Ollama(
            model=llm_model,
            temperature=0,
        ),
    }
    providers = get_providers()
    return llm_provider_map[providers["llms"][llm_model]]()


def get_index_name(collection_name: str):
    return (
        f"{os.environ['DB_DIRECTORY']}/{os.environ['VECTOR_DB_TYPE']}/{collection_name}"
    )


def load_db(embedding_model: str, collection_name: str, keys: Dict[str, str]):
    def load_db_chromadb(embeddings: Embeddings):
        return Chroma(
            persist_directory=get_index_name(collection_name=collection_name),
            embedding_function=embeddings,
        )

    def load_db_faiss(embeddings: Embeddings):
        return FAISS.load_local(
            folder_path=get_index_name(collection_name=collection_name),
            embeddings=embeddings,
        )

    db_map = {
        "chromadb": load_db_chromadb,
        "faiss": load_db_faiss,
    }
    return db_map[os.environ["VECTOR_DB_TYPE"]](
        embeddings=load_cached_embedding(embedding_model, keys=keys)
    )


def save_db(embedding_model: str, collection_name: str, documents: List[Document]):
    def save_db_chromadb(embeddings: Embeddings):
        db = Chroma.from_texts(
            texts=[t.page_content for t in documents],
            embedding=embeddings,
            persist_directory=get_index_name(collection_name=collection_name),
        )
        db.persist()

    def save_db_faiss(embeddings: Embeddings):
        FAISS.from_texts(
            texts=[t.page_content for t in documents],
            embedding=embeddings,
        ).save_local(get_index_name(collection_name=collection_name))

    db_map = {"chromadb": save_db_chromadb, "faiss": save_db_faiss}
    db_map[os.environ["VECTOR_DB_TYPE"]](
        embeddings=load_cached_embedding(embedding_model)
    )


def load_documents(docs_path):
    loader = PyPDFDirectoryLoader(docs_path)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    docs = text_splitter.split_documents(data)
    return docs


def ingestion(docs_path: str, embedding_model: str, collection_name: str):
    print("Loading documents...")
    docs = load_documents(docs_path)
    print("Adding documents to vector store...")
    save_db(
        embedding_model=embedding_model, collection_name=collection_name, documents=docs
    )
    print(f"{len(docs)} docs inserted into vector store {collection_name}")


def build_template():
    system_prompt = """Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer."""
    begin_instruction, end_instruction = "[INST]", "[/INST]"
    begin_system, end_system = "<>\n", "\n<>\n\n"
    system_prompt = begin_system + system_prompt + end_system
    instruction = """
        {context}

        Question: {question}
        """
    return begin_instruction + system_prompt + instruction + end_instruction


def run(
    chain_type: str,
    llm_model: str,
    embedding_model: str,
    collection_name: str,
    query: str,
    keys: Dict[str, str],
):
    prompt = PromptTemplate(
        template=build_template(), input_variables=["context", "question"]
    )

    db = load_db(
        embedding_model=embedding_model, collection_name=collection_name, keys=keys
    )
    llm = load_llm(llm_model=llm_model, keys=keys)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type=chain_type,
        retriever=db.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

    response = qa_chain({"query": query})
    return {"answer": response["result"], "documents": response["source_documents"]}


def load_data():
    # embeddings=bert(gpt4all)
    ingestion(docs_path="docs", embedding_model="bert", collection_name="gpt4all-bert")

    # embeddings=text-embedding-ada-002
    ingestion(
        docs_path="docs",
        embedding_model="text-embedding-ada-002",
        collection_name="openai-text-embedding-ada-002",
    )

    # embeddings=llama2
    ingestion(
        docs_path="docs", embedding_model="llama2", collection_name="ollama-llama2"
    )

    # embeddings=mistral
    ingestion(
        docs_path="docs", embedding_model="mistral", collection_name="ollama-mistral"
    )


def query_data(query: str):
    return run(
        chain_type="stuff",
        llm_model="gpt-3.5-turbo",
        embedding_model="text-embedding-ada-002",
        collection_name="openai-text-embedding-ada-002",
        query=query,
        keys={"OPENAI_API_KEY": os.environ["MY_OPENAI_API_KEY"]},
    )
    # return run(
    #     chain_type="stuff",
    #     llm_model="gpt-3.5-turbo",
    #     embedding_model="bert",
    #     collection_name="gpt4all-bert",
    #     query=query,
    #     keys={"OPENAI_API_KEY": os.environ["MY_OPENAI_API_KEY"]}
    # )
    # return run(
    #     chain_type="stuff",
    #     llm_model="llama2",
    #     embedding_model="llama2",
    #     collection_name="ollama-llama2",
    #     query=query,
    #     keys={"OPENAI_API_KEY": os.environ["MY_OPENAI_API_KEY"]}
    # )
    # return run(
    #     chain_type="stuff",
    #     llm_model="mistral",
    #     embedding_model="mistral",
    #     collection_name="ollama-mistral",
    #     query=query,
    #     keys={"OPENAI_API_KEY": os.environ["MY_OPENAI_API_KEY"]}
    # )


if __name__ == "__main__":
    load_data()
    # result = query_data(query="What is Yolov7?")
    # print(result["answer"])
