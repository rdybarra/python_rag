import textwrap
from pprint import pprint

import chromadb
import python_rag_common

DESC = textwrap.dedent(
    """\
This script queries embeddings using chromaDBs built-in embedding model.
It instantiates chromaDB as an ephemeral instance.

*  Where does the search corpus come from? Variables in the script
*  How does it create embeddings? Built-in ChromaDB capability
*  What model does it use for a response? None, just vector search
"""
)


def populate_and_query_chroma_embeddings(is_interactive=False):
    # This will use ChromaDBs embeddings. It performs better in my testing than the
    # ollama model nomic-embed-text that I have locally. However, it's still not
    # amazing.
    documents = [
        "This is a document about pineapple",
        "This is a document about oranges",
    ]

    chroma_client = chromadb.Client()
    embed_func = python_rag_common.get_chromadb_embedding_function()
    collection = chroma_client.create_collection(
        name="my_collection", embedding_function=embed_func
    )

    collection.add(documents=documents, ids=["id-pineapple", "id-oranges"])
    python_rag_common.print_collection(collection)

    default_query = "A question about most florida juice"
    print(f"\nExample: {default_query}")

    while True:
        query = input("\nQuery (or q/quit to quit): ") if is_interactive else default_query
        if query.lower() in ["q", "quit"]:
            break

        results = collection.query(
            query_texts=query,
            n_results=2,  # how many results to return
        )
        pprint(results)
        if not is_interactive:
            break


def main():
    parser = python_rag_common.init_parser(DESC)
    args = parser.parse_args()
    populate_and_query_chroma_embeddings(args.interactive)


if __name__ == "__main__":
    main()
