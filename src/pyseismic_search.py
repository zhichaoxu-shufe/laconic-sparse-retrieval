from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


def semantic_search_seismic(
    query_embeddings_decoded: list[list[tuple[str, float]]],
    corpus_embeddings_decoded: list[list[tuple[str, float]]] | None = None,
    corpus_index: tuple[SeismicIndex, str] | None = None,
    top_k: int = 10,
    output_index: bool = False,
    index_kwargs: dict[str, Any] | None = None,
    search_kwargs: dict[str, Any] | None = None,
) -> (
    tuple[list[list[dict[str, int | float]]], float]
    | tuple[list[list[dict[str, int | float]]], float, tuple[SeismicIndex, str]]
):
    """
    This function is from sentence_transformers
    https://github.com/huggingface/sentence-transformers/blob/main/sentence_transformers/sparse_encoder/search_engines.py
    currently deprecated in favor of the large vocabulary version below.

    Performs semantic search using sparse embeddings with Seismic.

    Args:
        query_embeddings_decoded: List of query embeddings in format [[("token": value), ...], ...]
            Example: To get this format from a SparseEncoder model::

                model = SparseEncoder('my-sparse-model')
                query_texts = ["your query text"]
                query_embeddings = model.encode(query_texts)
                query_embeddings_decoded = model.decode(query_embeddings)
        corpus_embeddings_decoded: List of corpus embeddings in format [[("token": value), ...], ...]
            Only used if corpus_index is None
            Can be obtained using the same decode method as query embeddings
        corpus_index: Tuple of (SeismicIndex, collection_name)
            If provided, uses this existing index for search
        top_k: Number of top results to retrieve
        output_index: Whether to return the SeismicIndex client and collection name
        index_kwargs: Additional arguments for SeismicIndex passed to build_from_dataset,
            such as centroid_fraction, min_cluster_size, summary_energy, nknn, knn_path,
            batched_indexing, or num_threads.
        search_kwargs: Additional arguments for SeismicIndex passed to batch_search,
            such as query_cut, heap_factor, n_knn, sorted, or num_threads.
            Note: query_cut and heap_factor are set to default values if not provided.
    Returns:
        A tuple containing:
        - List of search results in format [[{"corpus_id": int, "score": float}, ...], ...]
        - Time taken for search
        - (Optional) Tuple of (SeismicIndex, collection_name) if output_index is True
    """
    try:
        from seismic import SeismicDataset, SeismicIndex, get_seismic_string
    except ImportError:
        raise ImportError("Please install Seismic with `pip install pyseismic-lsr` to use this function.")

    if index_kwargs is None:
        index_kwargs = {}
    if search_kwargs is None:
        search_kwargs = {}

    string_type = get_seismic_string()

    # Validate input sparse tensors
    if not isinstance(query_embeddings_decoded, list) or not all(
        isinstance(item, list) and all(isinstance(t, tuple) and len(t) == 2 for t in item)
        for item in query_embeddings_decoded
    ):
        raise ValueError("Query embeddings must be a list of lists in the format [[('token', value), ...], ...]")

    if corpus_index is None:
        if corpus_embeddings_decoded is None:
            raise ValueError("Either corpus_embeddings_decoded or corpus_index must be provided")

        if not isinstance(corpus_embeddings_decoded, list) or not all(
            isinstance(item, list) and all(isinstance(t, tuple) and len(t) == 2 for t in item)
            for item in corpus_embeddings_decoded
        ):
            raise ValueError("Corpus embeddings must be a list of lists in the format [[('token', value), ...], ...]")

        # Create new Seismic dataset
        dataset = SeismicDataset()

        num_vectors = len(corpus_embeddings_decoded)

        # Add each document to the Seismic dataset
        for idx in tqdm(range(num_vectors), desc="Adding documents to Seismic"):
            tokens = dict(corpus_embeddings_decoded[idx])
            dataset.add_document(
                str(idx),
                np.array(list(tokens.keys()), dtype=string_type),
                np.array(list(tokens.values()), dtype=np.float32),
            )

        corpus_index = SeismicIndex.build_from_dataset(dataset, **index_kwargs)

    search_start_time = time.time()

    num_queries = len(query_embeddings_decoded)
    # Process indices and values for batch search
    query_components = []
    query_values = []

    # Create query components and values for each query
    for q_idx in range(num_queries):
        query_tokens = dict(query_embeddings_decoded[q_idx])
        query_components.append(np.array(list(query_tokens.keys()), dtype=string_type))
        query_values.append(np.array(list(query_tokens.values()), dtype=np.float32))

    if "query_cut" not in search_kwargs:
        search_kwargs["query_cut"] = 10
    if "heap_factor" not in search_kwargs:
        search_kwargs["heap_factor"] = 0.7
    results = corpus_index.batch_search(
        queries_ids=np.array(range(num_queries), dtype=string_type),
        query_components=query_components,
        query_values=query_values,
        k=top_k,
        **search_kwargs,
    )

    # Sort the results by query index
    results = sorted(results, key=lambda x: int(x[0][0]))

    # Format results
    all_results = [
        [{"corpus_id": int(corpus_id), "score": score} for query_idx, score, corpus_id in query_result]
        for query_result in results
    ]

    search_time = time.time() - search_start_time

    if output_index:
        return all_results, search_time, corpus_index
    else:
        return all_results, search_time
    

def semantic_search_seismic_large_vocabulary(
    query_embeddings_decoded: list[list[tuple[str, float]]],
    corpus_embeddings_decoded: list[list[tuple[str, float]]] | None = None,
    corpus_index: tuple[SeismicIndexLV, str] | None = None,
    encodings_path: str | None = None,
    top_k: int = 10,
    output_index: bool = False,
    index_kwargs: dict[str, Any] | None = None,
    search_kwargs: dict[str, Any] | None = None,
    ):
    """
    Performs semantic search using sparse embeddings with Seismic.

    Args:
        query_embeddings_decoded: List of query embeddings in format [[("token": value), ...], ...]
            Example: To get this format from a SparseEncoder model::

                model = SparseEncoder('my-sparse-model')
                query_texts = ["your query text"]
                query_embeddings = model.encode(query_texts)
                query_embeddings_decoded = model.decode(query_embeddings)
        corpus_embeddings_decoded: List of corpus embeddings in format [[("token": value), ...], ...]
            Only used if corpus_index is None
            Can be obtained using the same decode method as query embeddings
        encodings_path: Path to the merged corpus encodings file in JSONL format.
            first check if encodings_path is provided, then corpus_embeddings_decoded is used.
        corpus_index: Tuple of (SeismicIndex, collection_name)
            If provided, uses this existing index for search
        top_k: Number of top results to retrieve
        output_index: Whether to return the SeismicIndex client and collection name
        index_kwargs: Additional arguments for SeismicIndex passed to build_from_dataset,
            such as centroid_fraction, min_cluster_size, summary_energy, nknn, knn_path,
            batched_indexing, or num_threads.
        search_kwargs: Additional arguments for SeismicIndex passed to batch_search,
            such as query_cut, heap_factor, n_knn, sorted, or num_threads.
            Note: query_cut and heap_factor are set to default values if not provided.
    Returns:
        A tuple containing:
        - List of search results in format [[{"corpus_id": int, "score": float}, ...], ...]
        - Time taken for search
        - (Optional) Tuple of (SeismicIndex, collection_name) if output_index is True
    """
    
    
    try:
        from seismic import SeismicDatasetLV, SeismicIndexLV, get_seismic_string
    except ImportError:
        raise ImportError("Please install Seismic with `pip install pyseismic-lsr` to use this function.")

    if index_kwargs is None:
        index_kwargs = {}
    if search_kwargs is None:
        search_kwargs = {}


    string_type = get_seismic_string()

    # Validate input sparse tensors
    if not isinstance(query_embeddings_decoded, list) or not all(
        isinstance(item, list) and all(isinstance(t, tuple) and len(t) == 2 for t in item)
        for item in query_embeddings_decoded
    ):
        raise ValueError("Query embeddings must be a list of lists in the format [[('token', value), ...], ...]")

    if corpus_index is None:
        if encodings_path is not None:
            # Load corpus embeddings from the merged encodings file
            corpus_index = SeismicIndexLV.build(encodings_path, **index_kwargs)

        else:
            if corpus_embeddings_decoded is None:
                raise ValueError("Either corpus_embeddings_decoded or corpus_index must be provided")

            if not isinstance(corpus_embeddings_decoded, list) or not all(
                isinstance(item, list) and all(isinstance(t, tuple) and len(t) == 2 for t in item)
                for item in corpus_embeddings_decoded
            ):
                raise ValueError("Corpus embeddings must be a list of lists in the format [[('token', value), ...], ...]")

            # Create new Seismic dataset
            dataset = SeismicDatasetLV()

            num_vectors = len(corpus_embeddings_decoded)

            # Add each document to the Seismic dataset
            for idx in tqdm(range(num_vectors), desc="Adding documents to Seismic"):
                tokens = dict(corpus_embeddings_decoded[idx])
                dataset.add_document(
                    str(idx),
                    np.array(list(tokens.keys()), dtype=string_type),
                    np.array(list(tokens.values()), dtype=np.float32),
                )

            corpus_index = SeismicIndexLV.build_from_dataset(dataset, **index_kwargs)

    search_start_time = time.time()

    num_queries = len(query_embeddings_decoded)
    # Process indices and values for batch search
    query_components = []
    query_values = []

    # Create query components and values for each query
    for q_idx in range(num_queries):
        query_tokens = dict(query_embeddings_decoded[q_idx])
        query_components.append(np.array(list(query_tokens.keys()), dtype=string_type))
        query_values.append(np.array(list(query_tokens.values()), dtype=np.float32))

    if "query_cut" not in search_kwargs:
        search_kwargs["query_cut"] = 10
    if "heap_factor" not in search_kwargs:
        search_kwargs["heap_factor"] = 0.7
    results = corpus_index.batch_search(
        queries_ids=np.array(range(num_queries), dtype=string_type),
        query_components=query_components,
        query_values=query_values,
        k=top_k,
        **search_kwargs,
    )

    # Sort the results by query index
    results = sorted(results, key=lambda x: int(x[0][0]))

    # Format results
    all_results = [
        [{"corpus_id": str(corpus_id), "score": score} for query_idx, score, corpus_id in query_result]
        for query_result in results
    ]

    search_time = time.time() - search_start_time

    if output_index:
        return all_results, search_time, corpus_index
    else:
        return all_results, search_time
    