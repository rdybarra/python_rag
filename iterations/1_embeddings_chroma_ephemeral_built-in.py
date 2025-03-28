# This script queries embeddings using chromaDBs built-in embedding model.
# It instantiates chromaDB as an ephermal instance

# Where does the search corpus come from? - Variables in the script
# How does it create embeddings? - Built-in ChromaDB capability
# What model does it use for a response? - None, just vector search

import chromadb


def populate_and_query_chroma_embedings():
    # This will use ChromaDBs embeddings. It performs better in my testing than the
    # ollama model nomic-embed-text that I have locally. However, it's still not
    # amazing.
    documents = [
        "This is a document about pineapple",
        "This is a document about oranges",
    ]

    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(name="my_collection")

    collection.add(documents=documents, ids=["idpinapple", "idoranages"])

    results = collection.query(
        query_texts="A question about most florida juice",
        n_results=2,  # how many results to return
    )
    print(results)


def main():
    populate_and_query_chroma_embedings()


if __name__ == "__main__":
    main()
