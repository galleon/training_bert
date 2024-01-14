import sys
import argparse
import pandas as pd
from chromadb.utils import embedding_functions
from datasets import Dataset
import chromadb
import torch
import csv
from tqdm.auto import tqdm
from time import time


class JoliRag:
    def __init__(self, model_name=None):
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        client = chromadb.PersistentClient(path="./db")
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )
        self.collection = client.get_or_create_collection(
            "ioga_collection", embedding_function=ef, metadata={"hnsw:space": "cosine"}
        )

        count = self.collection.count()
        print(f"Index contains {count} documents")

    def create_index(self, file_path):
        df = pd.read_csv(file_path)
        df.dropna(inplace=True)
        df["id"] = df["id"].astype(str)

        data = Dataset.from_pandas(df)

        batch_size = 100  # number of embeddings to be generated simultaneously

        for i in tqdm(range(0, len(data), batch_size)):
            # find end of batch
            i_end = min(len(data), i + batch_size)
            # create batch
            batch = data[i:i_end]
            # embeds = self.ef(batch["text"])
            # upsert to ChromaDb
            self.collection.add(
                ids=batch["id"],
                metadatas=None,
                documents=batch["text"],
            )

    def search(self, query, top_k=10):
        assert (
            self.collection.count() > 0
        ), "Index is empty. Please index some documents first."

        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            #  where={"metadata_field": "is_equal_to_this"},
            #  where_document={"$contains":"search_string"}
        )
        return results


def main():
    parser = argparse.ArgumentParser(description="Text Embedding and Retrieval System")
    parser.add_argument(
        "--mode",
        choices=["index", "retrieve"],
        required=True,
        help="Mode of operation: 'index' to create index or 'retrieve' to search",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Name of the pre-trained model to use for embedding",
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Path to the input CSV file (for indexing) or index file (for retrieval)",
    )
    parser.add_argument(
        "--query", type=str, help="Query for retrieving similar text snippets"
    )
    parser.add_argument("--query_file", type=str, help="Path to the query CSV file")
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of similar text snippets to retrieve",
    )

    args = parser.parse_args()

    jr = JoliRag(model_name=args.model_name)

    if args.mode == "index":
        if not args.file:
            raise ValueError("Input CSV file path is required for indexing mode")
        jr.create_index(args.file)
    elif args.mode == "retrieve":
        if args.query:
            t1 = time()
            results = jr.search(args.query, args.top_k)
            print(f"Search time: {time() - t1}")
            print(results)


if __name__ == "__main__":
    main()