import os
import textwrap
import uuid
from pprint import pprint

import chromadb
import python_rag_common
from dotenv import load_dotenv
from google import genai
from langchain_text_splitters import RecursiveCharacterTextSplitter

DESC = textwrap.dedent(
    """\
This script creates and queries embeddings from a gemini embedding model and stores
them in a PERSISTED instance of ChromaDB.
It searches chromaDB to add context and then searches using gemini.

* Where does the search corpus come from? Text file
* How does it create embeddings? Gemini
* What model does it use for a response? Gemini
"""
)

EPILOG = textwrap.dedent(
    """\
Requirements
"GEMINI_API_KEY", "CHROMA_HOST", "CHROMA_PORT" set in ".env" file at root.
"""
)

PROMPT_TEMPLATE = """\
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def get_embeddings_for_input(input):
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=gemini_api_key)

    result = client.models.embed_content(model="text-embedding-004", contents=input)

    print(len(result.embeddings[0].values))
    return result.embeddings[0].values


def gemini_query(is_interactive=False):
    load_dotenv()
    # Open the file in read mode
    with open("data/escondido.txt", "r") as file:
        # Read the entire file content
        content = file.read()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_text(content)

    ids = []
    embeddings = []

    for text_chunk in texts:
        embeddings.append(get_embeddings_for_input(text_chunk))
        ids.append(str(uuid.uuid4()))

    # print(texts)

    chroma_host = os.environ.get("CHROMA_HOST")
    chroma_port = os.environ.get("CHROMA_PORT")
    chroma_client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
    collection = chroma_client.create_collection(name="5_gemini")

    print(len(ids))
    print(len(embeddings))

    collection.add(documents=texts, embeddings=embeddings, ids=ids)
    python_rag_common.print_collection(collection)
    print("collection added")

    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=gemini_api_key)

    default_query = "Who settled Escondido?"
    print(f"\nExample: {default_query}")

    while True:
        query = input("\nQuery (or q/quit to quit): ") if is_interactive else default_query
        if query.lower() in ["q", "quit"]:
            break

        results = collection.query(
            query_embeddings=get_embeddings_for_input(query),
            n_results=4,
        )
        pprint(results)

        formatted_prompt = PROMPT_TEMPLATE.format(context=str(results["documents"]), question=query)
        pprint(formatted_prompt)

        response = client.models.generate_content(
            model="gemini-2.0-flash", contents=[formatted_prompt]
        )
        print("ANSWER")
        print(response.text)
        if not is_interactive:
            break


def main():
    parser = python_rag_common.init_parser(DESC, EPILOG)
    args = parser.parse_args()
    gemini_query(args.interactive)


if __name__ == "__main__":
    main()
