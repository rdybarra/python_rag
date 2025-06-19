import textwrap
from pathlib import Path
from pprint import pprint

import chromadb
import PyPDF2
import python_rag_common
from langchain_ollama.llms import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter

DESC = textwrap.dedent(
    """\
Search a PDF, get an answer.

* Where does the search corpus come from? PDF
* How does it create embeddings? ChromaDB built-in capability
* What model does it use for a response? Local ollama (Deepseek:8b)
"""
)

PROMPT_TEMPLATE = """\
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def extract_text_from_pdf(pdf_path):
    # Open the PDF file in binary read mode
    with open(pdf_path, "rb") as file:
        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfReader(file)

        # Initialize a variable to store the extracted text
        extracted_text = ""

        # Iterate through each page in the PDF
        for page in pdf_reader.pages:
            # Extract text from the page and append it to the extracted_text variable
            extracted_text += page.extract_text()

    return extracted_text


def populate_and_query_chroma_embeddings(
    pdf_path, is_interactive=False, model_name="deepseek-r1:8b"
):
    # This will use ChromaDBs embeddings. It performs better in my testing than the
    # ollama model nomic-embed-text that I have locally. However, it's still not
    # amazing
    text = extract_text_from_pdf(pdf_path)
    print(text[:200])

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_text(text)

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

    default_query = "Where should I store my trash and recycle bins?"
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

    # THIS PDF IS NOT IN SOURCE CONTROL
    pdf_path = Path("data/hoa_canopy_grove.pdf")
    if not (pdf_path.exists() and pdf_path.is_file()):
        raise Exception(f"{pdf_path}: Is not an existing file")
    populate_and_query_chroma_embeddings(pdf_path, args.interactive, args.ollama_model)


if __name__ == "__main__":
    main()
