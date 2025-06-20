import textwrap
from pprint import pprint

import chromadb
import ollama
import python_rag_common

DESC = textwrap.dedent(
    """\
This script queries embeddings using a local ollama embedding model.
It instantiates chromaDB as an ephemeral instance

This requires that ollama is installed, running and the model is pulled down.
i.e.: the default model is "nomic-embed-text": https://ollama.com/library/nomic-embed-text

*  Where does the search corpus come from? Variables in the script
*  How does it create embeddings? Local ollama
*  What model does it use for a response? None, just vector search
"""
)


def get_embeddings_for_input(input, model):
    response = ollama.embed(model=model, input=input)

    return response.embeddings[0]


def populate_and_query_ollama_embeddings(is_interactive=False, model="nomic-embed-text"):
    # This function gets embeddings from a local ollama model (nomic-embed-text). It doesn't
    # appear to be as good as ChromaDB's built in embeddings.

    documents = [
        "This is a document about pineapple",
        "This is a document about oranges",
    ]

    ids = ["id-pineapple", "id-oranges"]
    embeddings = [get_embeddings_for_input(doc, model) for doc in documents]

    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(name="my_collection")

    # I prefer passing embeddings instead of the embedding functions because this way demonstrates
    # a greater separation of concerns. Perhaps there are preformance reasons not to do this in
    # production code?
    collection.add(documents=documents, embeddings=embeddings, ids=ids)

    default_query = "This is a query about spikey hawaiian fruit"
    print(f"\nExample: {default_query}")

    while True:
        query = (
            input(f"\nQuery [{model}] (or q/quit to quit): ") if is_interactive else default_query
        )
        if query.lower() in ["q", "quit"]:
            break

        embeddings_for_query = get_embeddings_for_input(query, model)

        results = collection.query(
            query_embeddings=embeddings_for_query,
            n_results=2,  # how many results to return
        )
        pprint(results)
        if not is_interactive:
            break


def main():
    parser = python_rag_common.init_parser(DESC)
    parser.add_argument("--ollama-model", default="nomic-embed-text", help="Ollama model to use")
    args = parser.parse_args()
    populate_and_query_ollama_embeddings(args.interactive, args.ollama_model)


if __name__ == "__main__":
    main()
