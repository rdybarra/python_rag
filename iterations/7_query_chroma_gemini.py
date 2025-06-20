# Search an existing corpus and get an answer

# Where does the search corpus come from? - Previusly created embeddings
# How does it create embeddings? - Gemini (only search term embeddings are needed)
# What model does it use for a response? - Gemini

# "GEMINI_API_KEY", "CHROMA_HOST", "CHROMA_PORT" set in ".env" file at root.

import os

import chromadb
from dotenv import load_dotenv
from google import genai


def get_embeddings_for_input(input):
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=gemini_api_key)

    result = client.models.embed_content(model="text-embedding-004", contents=input)

    print(len(result.embeddings[0].values))
    return result.embeddings[0].values


def gemini_query():
    load_dotenv()
    chroma_host = os.environ.get("CHROMA_HOST")
    chroma_port = os.environ.get("CHROMA_PORT")
    chroma_client = chromadb.HttpClient(host=chroma_host, port=chroma_port)

    print("getting collection")

    collection = chroma_client.get_collection(name="5_gemini")

    print("collection found")

    question = "Who settled Escondido?"

    results = collection.query(
        query_embeddings=get_embeddings_for_input(question),
        n_results=4,
    )
    print(results)
    print("done")

    PROMPT_TEMPLATE = """
    Answer the question based only on the following context:

    {context}

    ---

    Answer the question based on the above context: {question}
    """

    formatted_prompt = PROMPT_TEMPLATE.format(context=str(results["documents"]), question=question)
    print(formatted_prompt)

    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=gemini_api_key)

    response = client.models.generate_content(model="gemini-2.0-flash", contents=[formatted_prompt])
    print("ANSWER")
    print(response.text)


def main():
    gemini_query()


if __name__ == "__main__":
    main()
