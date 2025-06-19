import textwrap
from pprint import pprint

import chromadb
import python_rag_common
from langchain_ollama.llms import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter

DESC = textwrap.dedent(
    """\
This script queries embeddings using chromaDBs built-in embedding model and then
calls a local ollama model to get an answer.
In this case I'm defaulting to deepseek's 8b parameter model: https://ollama.com/library/deepseek-r1
The embeddings come from an external text file.
It instantiates chromaDB as an ephemeral instance

*  Where does the search corpus come from? Text file
*  How does it create embeddings? ChromaDBs built-in capability
*  What model does it use for a response? Local ollama (deepseek:8b)
"""
)

PROMPT_TEMPLATE = """\
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def populate_and_query_chroma_embeddings(is_interactive=False, model_name="deepseek-r1:8b"):
    # This will use ChromaDBs embeddings. It performs better in my testing than the
    # ollama model nomic-embed-text that I have locally. However, it's still not
    # amazing

    # Open the file in read mode
    with open("data/escondido.txt", "r") as file:
        # Read the entire file content
        content = file.read()

    # This is used to split up large blocks of texts into more manageable chunks.
    # This tool is nice as it allows many different splitting strategies and will do
    # nice things like try to keep paragraphs together. You could easily implement on
    # your own but this is handy.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_text(content)

    # This is how the cool kids do it I guess? This syntax will take some getting used to.
    # texts_with_ids = [(i, segment) for i, segment in enumerate(texts, start=1)]

    ids = []
    for i, _ in enumerate(texts, start=1):
        ids.append(str(i))

    # print(texts)

    chroma_client = chromadb.Client()
    embed_func = python_rag_common.get_chromadb_embedding_function()
    collection = chroma_client.create_collection(
        name="my_collection", embedding_function=embed_func
    )

    collection.add(documents=texts, ids=ids)
    python_rag_common.print_collection(collection)
    model = OllamaLLM(model=model_name)

    default_query = "Who settled Escondido?"
    print(f"\nExample: {default_query}")

    while True:
        query = (
            input(f"\nQuery [{model_name}] (or q/quit to quit): ")
            if is_interactive
            else default_query
        )
        if query.lower() in ["q", "quit"]:
            break

        results = collection.query(
            query_texts=query,
            n_results=4,
        )
        pprint(results)

        formatted_prompt = PROMPT_TEMPLATE.format(context=str(results["documents"]), question=query)
        print(formatted_prompt)

        response_text = model.invoke(formatted_prompt)
        print("ANSWER")
        print(response_text)
        if not is_interactive:
            break


def main():
    parser = python_rag_common.init_parser(DESC)
    parser.add_argument("--ollama-model", default="deepseek-r1:8b", help="Ollama model to use")
    args = parser.parse_args()
    populate_and_query_chroma_embeddings(args.interactive, args.ollama_model)


if __name__ == "__main__":
    main()
