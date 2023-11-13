# chatbot

🤖 Chatbot made with Streamlit using multiple LLMs, embedding models and databases.

This project mostly uses free services/images/models to create a chatbot.
You can still use OpenAI if you want to.

Stack:

- **Embeddings**
  - [Paid] text-embedding-ada-002
  - [Free] llama2
  - [Free] mistral
  - [Free] bert (from GPT4All)
- **LLMs**
  - [Paid] gpt-3.5-turbo
  - [Paid] gpt-4
  - [Free] llama2
  - [Free] mistral
- **Vector Stores**
  - [Free] ChromaDB
  - [Free] FAISS
- **Libs**
  - Langchain
  - Ollama
- **Cache**
  - Redis
- **Containers**
  - Docker
  - Docker Compose
- **Package Manager**
  - Poetry

## Setup

- 1.Add all your pdf files inside `/docs` folder
- 2.Create a folder to store models download by Ollama and its configs. By default it will look into `%USERPROFILE%\.docker\custom-volumes\ollama`. If you're using Linux or Mac replace the `${USERPROFILE}` by `${HOME}` in `docker-compose.yml`
- 3.Run `docker compose up -d`.
- 4.Inside a virtual env, install the deps: `poetry install`
- 5.Create `.env` file (example in next section)
- 6.Run `streamlit run ui.py`

### .env file

```env
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_NAMESPACE=embedding_caches
REDIS_DB=0
VECTOR_DB_TYPE=chromadb
DB_DIRECTORY=local/data/db
```

## Cache

Cache is only used for embeddings using **Redis** as backend.
This is only to make easier switch between embeddings models to compare the results.

## Vector Store

I'm using **ChromaDB** and **FAISS** as vector stores, but you can switch it to anything supported by langchain, like Pinecone, PgVector, etc.

## To Do

- [ ] Create UI for data ingestion
- [ ] Allow user to choose the database
- [ ] Show documents and scores returned from similarity search
