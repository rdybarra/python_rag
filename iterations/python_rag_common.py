import argparse
import platform

from chromadb.utils import embedding_functions
from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2


def get_chromadb_embedding_function():
    """Intel Mac work-around for chroma ONNXRuntimeError.

    See: https://github.com/chroma-core/chroma/issues/2731"""
    ef = embedding_functions.DefaultEmbeddingFunction()
    if platform.system() == "Darwin" and platform.processor() == "i386":
        ef = ONNXMiniLM_L6_V2(preferred_providers=["CPUExecutionProvider"])
    return ef


def init_parser(desc, epilog=None):
    """Reusable arg parser that supports --interactive flag"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=desc, epilog=epilog
    )
    parser.add_argument(
        "-i", "--interactive", default=False, action="store_true", help="prompt user for query"
    )
    return parser


def print_collection(collection):
    """Helper method that prints details of ChromaDB collection"""
    result = collection.peek()
    count = collection.count()
    # not sure why .peek() occasionally returns more results than .count() does...
    limit = min(len(result), count)
    print(f"Found {count} embeddings in ChromaDB. Showing first {limit}")
    for i in range(0, limit):
        print(f"{i}: id={result["ids"][i]:<15} doc={result["documents"][i]}")
