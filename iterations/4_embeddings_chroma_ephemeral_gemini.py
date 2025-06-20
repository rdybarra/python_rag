import os
import textwrap
from pprint import pprint

import chromadb
import python_rag_common
from dotenv import load_dotenv
from google import genai

DESC = textwrap.dedent(
    """\
This script does embedding search with Google Gemini.
It uses an ephemeral instance of ChromaDB

* Where does the search corpus come from? Variables in the script
* How does it create embeddings? Gemini
* What model does it use for a response? Nothing, just vector search
"""
)

EPILOG = textwrap.dedent(
    """\
Requirements
"GEMINI_API_KEY" set in ".env" file at root.
"""
)


def get_embeddings_for_input(input):
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=gemini_api_key)

    result = client.models.embed_content(model="text-embedding-004", contents=input)

    print(result.embeddings[0].values)
    return result.embeddings[0].values


def populate_and_query_gemini_embeddings(is_interactive=False):
    # This function gets embeddings from gemini.
    load_dotenv()

    documents = [
        "This is a document about pineapple",
        "This is a document about oranges",
    ]

    ids = ["id-pineapple", "id-oranges"]
    embeddings = [get_embeddings_for_input(doc) for doc in documents]

    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(name="my_collection")

    # I prefer passing embeddings instead of the embedding functions because this way demonstrates
    # a greater separation of concerns. Perhaps there are preformance reasons not to do this in
    # production code?
    collection.add(documents=documents, embeddings=embeddings, ids=ids)
    python_rag_common.print_collection(collection)

    default_query = "This is a query about popular florida orchard"
    print(f"\nExample: {default_query}")

    while True:
        query = input("\nQuery (or q/quit to quit): ") if is_interactive else default_query
        if query.lower() in ["q", "quit"]:
            break
        embeddings_for_query = get_embeddings_for_input(query)

        results = collection.query(
            query_embeddings=embeddings_for_query,
            n_results=2,  # how many results to return
        )
        pprint(results)
        if not is_interactive:
            break


def main():
    parser = python_rag_common.init_parser(DESC, EPILOG)
    args = parser.parse_args()
    populate_and_query_gemini_embeddings(args.interactive)


if __name__ == "__main__":
    main()
