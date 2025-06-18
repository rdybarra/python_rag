# This script queries embeddings using a local ollama embedding model.
# This requires that ollama be installed and running and that the "nomic-embed-text"
# model is pulled down. https://ollama.com/library/nomic-embed-text
# It instantiates chromaDB as an ephermal instance

# Where does the search corpus come from? - Variables in the script
# How does it create embeddings? - Local ollama
# What model does it use for a response? - None, just vector search

import chromadb
import ollama


def get_embeddings_for_input(input):
    response = ollama.embed(model="nomic-embed-text", input=input)

    return response.embeddings[0]


def populate_and_query_ollama_embeddings():
    # This function gets embeddings from a local ollama model (nomic-embed-text). It doesn't
    # appear to be as good as ChromaDB's built in embeddings.

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

    embeddings_for_query = get_embeddings_for_input("This is a query about spikey hawaiian fruit")

    results = collection.query(
        query_embeddings=embeddings_for_query,
        n_results=2,  # how many results to return
    )
    print(results)


def main():
    populate_and_query_ollama_embeddings()


if __name__ == "__main__":
    main()
