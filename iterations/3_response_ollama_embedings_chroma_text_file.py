# This script queries embeddings using chromaDBs built-in embedding model and then
# calls a local ollama model to get an answer.
# In this case I'm using deepseek's 8b parameter model: https://ollama.com/library/deepseek-r1
# The embeddings come from an external text file.
# It instantiates chromaDB as an ephermal instance

# Where does the search corpus come from? - Text file
# How does it create embeddings? - ChromaDBs built-in capability
# What model does it use for a response? - Local ollama (deepseek:8b)

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms.ollama import Ollama


def populate_and_query_chroma_embedings():
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
    collection = chroma_client.create_collection(name="my_collection")

    collection.add(documents=texts, ids=ids)

    question = "Who settled Escondido?"

    results = collection.query(
        query_texts=question,
        n_results=4,
    )
    print(results)

    PROMPT_TEMPLATE = """
    Answer the question based only on the following context:

    {context}

    ---

    Answer the question based on the above context: {question}
    """

    formatted_prompt = PROMPT_TEMPLATE.format(
        context=str(results["documents"]), question=question
    )
    print(formatted_prompt)

    model = Ollama(model="deepseek-r1:8b")
    response_text = model.invoke(formatted_prompt)

    print("ANSWER")
    print(response_text)


def main():
    populate_and_query_chroma_embedings()


if __name__ == "__main__":
    main()
