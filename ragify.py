import argparse
import os
from time import time

import chromadb
import pandas as pd
import torch
from datasets import Dataset
from llama_index import ServiceContext, VectorStoreIndex
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.retrievers import BM25Retriever, QueryFusionRetriever
from llama_index.schema import TextNode
from llama_index.storage.docstore import SimpleDocumentStore
from llama_index.storage.storage_context import StorageContext, DEFAULT_PERSIST_DIR
from llama_index.vector_stores import ChromaVectorStore
from tqdm.auto import tqdm


# class HybridRetriever(BaseRetriever):
#     def __init__(self, vector_retriever, bm25_retriever):
#         self.vector_retriever = vector_retriever
#         self.bm25_retriever = bm25_retriever
#         super().__init__()

#     def _retrieve(self, query, **kwargs):
#         bm25_nodes = self.bm25_retriever.retrieve(query, **kwargs)
#         vector_nodes = self.vector_retriever.retrieve(query, **kwargs)

#         # combine the two lists of nodes
#         all_nodes = []
#         node_ids = set()
#         for n in bm25_nodes + vector_nodes:
#             if n.node.node_id not in node_ids:
#                 all_nodes.append(n)
#                 node_ids.add(n.node.node_id)
#         return all_nodes


class ContentIndexerRetriever:
    def __init__(
        self, model_name=None, embed_batch_size=10, persist_dir=DEFAULT_PERSIST_DIR
    ):
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        self._persist_dir = persist_dir

        # Create a service context with a HuggingFaceEmbedding

        embed_model = HuggingFaceEmbedding(
            model_name=model_name, embed_batch_size=embed_batch_size, device=device
        )

        self._service_context = ServiceContext.from_defaults(
            embed_model=embed_model, llm=None
        )

        # Create a ChromaDB vector store

        client = chromadb.PersistentClient(path=persist_dir)

        chroma_collection = client.get_or_create_collection(
            f"ioga_collection_{model_name.replace('/', '_')}"
        )

        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        # Create a simple document store

        try:
            docstore = SimpleDocumentStore.from_persist_dir(persist_dir=persist_dir)
            print("Loaded existing docstore")
        except FileNotFoundError:
            docstore = SimpleDocumentStore()
            print("Created new docstore")

        # Create a storage context

        self._storage_context = StorageContext.from_defaults(
            docstore=docstore, vector_store=vector_store
        )

        # Load the index from storage

        if os.path.isfile(persist_dir + "/index_store.json"):
            self._index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store, service_context=self._service_context
            )

    def index_utterances_from_csv(self, filename):
        df = pd.read_csv(filename)
        df.dropna(inplace=True)
        df["id"] = df["id"].astype(str)

        data = Dataset.from_pandas(df)

        nodes = []
        for i in tqdm(range(0, len(data))):
            node = TextNode(
                text=data[i]["text"],
                id_=data[i]["id"],
                extra_info={"session_id": data[i]["id"]},
            )
            nodes.append(node)

        self._storage_context.docstore.add_documents(nodes)
        self._storage_context.docstore.persist()

        # self.vector_index = VectorStoreIndex.from_documents(documents, service_context=self._service_context, show_progress=True)
        t1 = time()
        self._index = VectorStoreIndex(
            nodes=nodes,
            storage_context=self._storage_context,
            service_context=self._service_context,
            show_progress=True,
        )

        self._index.storage_context.persist(persist_dir=self._persist_dir)

        print(f"Indexing time: {time() - t1:.2f} seconds")

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

    def report_metrics(self, similarity_file=None):
        assert (
            self._storage_context.vector_store.client.count() > 0
        ), "Index is empty. Please index some documents first."
        assert similarity_file, "Similarity file is required for reporting metrics"

        print("report_metrics")

        df = pd.read_csv(similarity_file)
        df.dropna(inplace=True)
        source_similarity_mapping = (
            df.groupby("SourceSessionId")["SimilaritySessionId"].apply(list).to_dict()
        )

        df_dict = {}
        df_dict["id"] = []
        for _top_k in [2, 10]:
            df_dict[f"top_{_top_k}"] = []

            vector_retriever = self._index.as_retriever(similarity_top_k=_top_k)

            t1 = time()
            bm25_retriever = BM25Retriever.from_defaults(
                docstore=self._storage_context.docstore, similarity_top_k=_top_k
            )
            print(f"BM25Retriever creation time: {time() - t1:.2f} seconds")

            for k, v in source_similarity_mapping.items():
                try:
                    query_text = self._storage_context.docstore.get_document(
                        str(k)
                    ).text
                    print(f"{k}: {len(query_text)} - {query_text}")

                    # will retrieve context from specific companies
                    nodes = bm25_retriever.retrieve(query_text)
                    set1 = set([node.node_id for node in nodes])

                    nodes = vector_retriever.retrieve(query_text)
                    set2 = set([node.node_id for node in nodes])

                    print(f"Overlap: {len(set1 & set2)/len(set1 | set2):.2%}")

                    if k not in df_dict["id"]:
                        df_dict["id"].append(k)

                    retrieved = (set1 | set2) & set([str(x) for x in v])
                    df_dict[f"top_{_top_k}"].append(len(retrieved) / len(v))
                except Exception as _:
                    continue
        return df_dict


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
        default="intfloat/multilingual-e5-small",  # "sentence-transformers/all-MiniLM-L6-v2"
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

    cir = ContentIndexerRetriever(model_name=args.model_name)

    if args.mode == "index":
        if not args.file:
            raise ValueError("Input CSV file path is required for indexing mode")
        cir.index_utterances_from_csv(args.file)
    elif args.mode == "retrieve":
        if args.query:
            t1 = time()
            results = cir.search(args.query, args.top_k)
            print(f"Search time: {time() - t1}")
            print(results)
        elif args.query_file:
            df_dict = cir.report_metrics(args.query_file)
            df = pd.DataFrame(df_dict)
            df.to_csv("ioga_metrics.csv", index=False)


if __name__ == "__main__":
    main()
