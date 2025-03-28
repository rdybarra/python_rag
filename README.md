# Explorations on using LLMs

Original inspiration is from [a RAG tutorial](https://github.com/pixegami/rag-tutorial-v2) but I felt like that code was too complete.

Various approaches are attempted in the "iterations" directory.

# High-Level flow of RAG

1. Get text from some source material.
2. Push the embedding representation of that text into a vector database (in small chunks)
3. Take a search term and search the vector database with its embedding representation.
4. Insert the results of the search into the langauge model prompt.
5. Ask your question to the lanaguage model and get results.

# Technologies used

The technologies are different for each script but I'm using

- [ChromaDB](https://www.trychroma.com) - For all vector databases

  - Most uses are ephemeral instances of ChromaDB
  - For the persistant uses of ChromaDB I followed these instructisons (https://docs.trychroma.com/production/cloud-providers/aws)

- [Ollama](https://ollama.com) - For any "local" models
- [GeminiSDK](https://ai.google.dev/gemini-api/docs) - Google's SDK for their models.
  - You will need to get your own API key to use these.

# Here are the scripts

1. Search embeddings with ChromaDB
2. Search embeddings with ChromaDB using a local embedding model
3. Import and split a text file to use as embeddings with ChromaDB and use a local model to get a response.
4. Practice embedding generation/search with Gemini SDK
5. Response and embedding search with Gemini SDK
6. Response and embedding search with Gemini SDK (Persistent Chroma instance)
7. Response and embedding search without creating new embeddings first.
8. Response and embedding search from PDF

# Pre-reqs

1. Some of the later scripts require API Keys and couple require URLs for ChromaDB. Those
   should live in a `.env` file.
2. The last script references a PDF that isn't there. You can use your own PDF and ask your own question.

# How to run

##

Run python 3.13.1 (I use https://github.com/pyenv/pyenv)

```
pyenv local 3.13.1
```

## Create a venv

```
python -m venv .venv
```

## Activate the venv

```
source .venv/bin/activate
```

## Install dependencies

```
pip install -r requirements.txt
```

## Execute the script

```
python iterations/<script name>
```
