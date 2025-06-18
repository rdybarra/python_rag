# This script does embedding search with Google Gemini.
# It uses an ephemeral instance of ChromaDB

# Where does the search corpus come from? - Variables in the script
# How does it create embeddings? - Gemini
# What model does it use for a response? - Nothing, just vector search

# Requirements
# "GEMINI_API_KEY" set in ".env" file at root.

import os

import chromadb
from dotenv import load_dotenv
from google import genai


def get_embeddings_for_input(input):
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=gemini_api_key)

    result = client.models.embed_content(model="text-embedding-004", contents=input)

    print(result.embeddings[0].values)
    return result.embeddings[0].values


def populate_and_query_gemini_embeddings():
    # This function gets embeddings from gemini.
    load_dotenv()

    documents = [
        "This is a document about pineapple",
        "This is a document about oranges",
    ]

    ids = ["id-pinapple", "id-oranges"]
    embeddings = []

    for doc in documents:
        embeddings.append(get_embeddings_for_input(doc))

    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(name="my_collection")

    # I prefer passing embeddings instead of the embedding functions because this way demonstrates
    # a greater separation of concerns. Perhaps there are preformance reasons not to do this in
    # production code?
    collection.add(documents=documents, embeddings=embeddings, ids=ids)

    embeddings_for_query = get_embeddings_for_input("This is a query about popular florida orchard")

    results = collection.query(
        query_embeddings=embeddings_for_query,
        n_results=2,  # how many results to return
    )
    print(results)


def main():
    populate_and_query_gemini_embeddings()


if __name__ == "__main__":
    main()
