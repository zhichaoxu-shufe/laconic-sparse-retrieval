#!/usr/bin/env python3
"""
Retrieve using pre-encoded sparse embeddings with PySeismic.

This script loads corpus and query embeddings from disk (encoded separately),
builds a PySeismic index, and performs retrieval.
"""

import argparse
import json
import orjson
import logging
import os
import pickle
import time
from pathlib import Path
from typing import List, Dict, Tuple
from multiprocessing import Pool, cpu_count

from tqdm import tqdm
from pyseismic_search import semantic_search_seismic_large_vocabulary

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_single_corpus_file(path: str) -> Tuple[List[List[Tuple[str, float]]], List[str]]:
    """
    Load a single corpus file and convert directly to seismic format.
    This function is designed to be called in parallel.
    
    Args:
        path: Path to .jsonl file containing corpus embeddings
        
    Returns:
        embeddings_seismic: List of embeddings in seismic format [(token, weight), ...]
        doc_ids: List of document IDs
    """
    embeddings_seismic = []
    doc_ids = []
    
    # Use larger I/O buffer
    with open(path, 'r', buffering=1024*1024) as f:
        batch_embeddings = []
        batch_ids = []
        
        for line in tqdm(f):
            line = line.strip()
            if not line:
                continue
            
            try:
                # Fast JSON parsing
                doc = orjson.loads(line)
                
                # Direct conversion (already optimized)
                seismic_vector = [(k, float(v)) for k, v in doc['vector'].items()]
                batch_embeddings.append(seismic_vector)
                batch_ids.append(doc['id'])
                
                # Batch extend every 1000 docs
                if len(batch_embeddings) >= 1000:
                    embeddings_seismic.extend(batch_embeddings)
                    doc_ids.extend(batch_ids)
                    batch_embeddings = []
                    batch_ids = []
                    
            except Exception as e:
                continue
        
        # Flush remaining
        if batch_embeddings:
            embeddings_seismic.extend(batch_embeddings)
            doc_ids.extend(batch_ids)

    return embeddings_seismic, doc_ids


def load_corpus_embeddings_parallel(corpus_paths: List[str], num_workers: int = None) -> Tuple[List[List[Tuple[str, float]]], List[str]]:
    """
    Load corpus embeddings from multiple .jsonl files in parallel.
    Directly converts to seismic format to reduce memory usage.
    
    Args:
        corpus_paths: List of paths to .jsonl files containing corpus embeddings
        num_workers: Number of parallel workers (default: min(#files, #CPUs))
        
    Returns:
        corpus_embeddings_seismic: List of embeddings in seismic format
        corpus_ids: List of document IDs
    """
    if num_workers is None:
        num_workers = min(len(corpus_paths), cpu_count() or 8)
    
    logger.info(f"Loading corpus from {len(corpus_paths)} file(s) using {num_workers} workers...")
    
    start_time = time.time()
    
    # Use multiprocessing to load files in parallel
    all_embeddings = []
    all_ids = []

    with Pool(processes=num_workers) as pool:
        for embeddings, ids in tqdm(
            pool.imap(load_single_corpus_file, corpus_paths),
            total=len(corpus_paths),
            desc="Loading corpus files"
        ):
            all_embeddings.extend(embeddings)
            all_ids.extend(ids)
            # Each worker result is processed and discarded immediately
            del embeddings, ids  # garbage collection
    
    elapsed = time.time() - start_time
    logger.info(f"Loaded {len(all_embeddings)} documents in {elapsed:.2f} seconds")
    
    return all_embeddings, all_ids


def load_corpus_embeddings(corpus_paths: List[str]) -> Tuple[List[Dict[str, float]], List[str]]:
    """
    Load corpus embeddings from one or more .jsonl files.
    
    Args:
        corpus_paths: List of paths to .jsonl files containing corpus embeddings
        
    Returns:
        corpus_embeddings: List of dicts mapping token to weight
        corpus_ids: List of document IDs
    """
    corpus_embeddings = []
    corpus_ids = []
    
    logger.info(f"Loading corpus from {len(corpus_paths)} file(s)...")
    
    for path in tqdm(corpus_paths, desc="Loading corpus files"):
        if not os.path.exists(path):
            logger.warning(f"Corpus file not found: {path}")
            continue
            
        with open(path, 'r') as f:
            for line in tqdm(f):
                try:
                    doc = json.loads(line.strip())
                    # Convert integer weights to float for seismic
                    vector = {k: float(v) for k, v in doc['vector'].items()}
                    corpus_embeddings.append(vector)
                    corpus_ids.append(doc['id'])
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line in {path}: {e}")
                    continue
    
    logger.info(f"Loaded {len(corpus_embeddings)} documents")
    return corpus_embeddings, corpus_ids


def load_query_embeddings(query_path: str) -> Tuple[List[List[Tuple[str, float]]], List[str]]:
    """
    Load query embeddings from .tsv file.
    Directly converts to seismic format to reduce memory usage.
    
    Supports two formats:
    1. JSON dict format (float weights): qid\t{"token1": 0.12, "token2": 0.08, ...}
    2. Repeated tokens format (int weights): qid\ttoken1 token1 token2 token2 token2 ...
    
    Args:
        query_path: Path to .tsv file containing query embeddings
        
    Returns:
        query_embeddings_seismic: List of embeddings in seismic format [(token, weight), ...]
        query_ids: List of query IDs
    """
    query_embeddings = []
    query_ids = []
    
    logger.info(f"Loading queries from {query_path}...")
    
    if not os.path.exists(query_path):
        raise FileNotFoundError(f"Query file not found: {query_path}")
    
    # Detect format from first line
    query_format = None
    with open(query_path, 'r') as f:
        first_line = f.readline().strip()
        if first_line:
            parts = first_line.split('\t')
            if len(parts) == 2:
                if parts[1].startswith('{'):
                    query_format = 'json'
                    logger.info("Detected JSON dict query format (float weights)")
                else:
                    query_format = 'repeated'
                    logger.info("Detected repeated tokens query format (int weights)")
    
    if query_format is None:
        raise ValueError("Could not detect query format from file")
    
    with open(query_path, 'r') as f:
        for line in tqdm(f, desc="Loading queries"):
            line = line.strip()
            if not line:
                continue
                
            try:
                parts = line.split('\t')
                if len(parts) != 2:
                    logger.warning(f"Invalid line format: {line[:50]}...")
                    continue
                
                qid, tokens_or_json = parts
                
                if query_format == 'json':
                    # JSON dict format - parse and convert directly to seismic
                    vector_dict = json.loads(tokens_or_json)
                    seismic_vector = [(k, float(v)) for k, v in vector_dict.items()]
                else:
                    # Repeated tokens format - count frequencies and convert to seismic
                    token_counts = {}
                    for token in tokens_or_json.split():
                        token_counts[token] = token_counts.get(token, 0) + 1
                    
                    seismic_vector = [(k, float(v)) for k, v in token_counts.items()]
                
                query_embeddings.append(seismic_vector)
                query_ids.append(qid)
            except Exception as e:
                logger.warning(f"Failed to parse query line: {e}")
                continue
    
    logger.info(f"Loaded {len(query_embeddings)} queries")
    return query_embeddings, query_ids


def write_rank_file(results: List[List[Dict]], query_ids: List[str], output_path: str):
    """
    Write retrieval results to TSV file.
    
    Format: qid\tpid\trank
    
    Args:
        results: List of retrieval results per query
        query_ids: List of query IDs
        output_path: Path to output TSV file
    """
    logger.info(f"Writing rank file to {output_path}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        for qid, result in zip(query_ids, results):
            for rank, entry in enumerate(result, start=1):
                pid = entry['corpus_id']
                f.write(f"{qid}\t{pid}\t{rank}\n")
    
    logger.info(f"Rank file written successfully with {len(query_ids)} queries")


def main():
    parser = argparse.ArgumentParser(
        description="Retrieve using pre-encoded sparse embeddings with PySeismic"
    )
    parser.add_argument(
        '--corpus_files',
        type=str,
        nargs='+',
        required=True,
        help='Path(s) to corpus .jsonl file(s). Can specify multiple files for sharded corpora.'
    )
    parser.add_argument(
        '--query_file',
        type=str,
        required=True,
        help='Path to query .tsv file'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        required=True,
        help='Path to output rank file (.tsv)'
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=100,
        help='Number of documents to retrieve per query (default: 100)'
    )
    parser.add_argument(
        '--index_cache',
        type=str,
        default=None,
        help='Path to save/load PySeismic index cache (.pkl). If exists, will load cached index.'
    )
    parser.add_argument(
        '--force_rebuild_index',
        action='store_true',
        help='Force rebuild index even if cache exists'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=None,
        help='Number of parallel workers for loading corpus files (default: auto-detect based on file count and CPU count)'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("PySeismic Retrieval from Pre-encoded Files")
    logger.info("=" * 60)
    logger.info(f"Corpus files: {args.corpus_files}")
    logger.info(f"Query file: {args.query_file}")
    logger.info(f"Output path: {args.output_path}")
    logger.info(f"Top-K: {args.top_k}")
    if args.index_cache:
        logger.info(f"Index cache: {args.index_cache}")
    logger.info("=" * 60)
    
    # Handle index caching
    corpus_index = None
    index_loaded = False
    
    if args.index_cache and os.path.exists(args.index_cache) and not args.force_rebuild_index:
        logger.info(f"Loading cached index from {args.index_cache}...")
        try:
            start_time = time.time()
            with open(args.index_cache, 'rb') as f:
                corpus_index = pickle.load(f)
            logger.info(f"Index loaded in {time.time() - start_time:.2f} seconds")
            index_loaded = True
        except Exception as e:
            logger.warning(f"Failed to load cached index: {e}")
            logger.info("Will rebuild index from scratch")
            corpus_index = None
    
    # Load corpus embeddings (skip if index is cached)
    # first check if the encodings is merged, if so, we can directly use seismic native index builder
    corpus_embeddings_seismic = None
    corpus_ids = None
    encodings_path = None
    if not index_loaded:
        if "merged" in args.corpus_files[0]:
            encodings_path = args.corpus_files[0]
            logger.info(f"Detected merged encodings file: {encodings_path}")

            # update: seismic can directly build index from merged encodings
            # thus we skip loading corpus_ids
            # with open(os.path.join(os.path.dirname(encodings_path), "corpus_ids.txt"), 'r') as f:
            #     corpus_ids = [line.strip() for line in f]

        else:
            start_time = time.time()
            if len(args.corpus_files) > 1:
                corpus_embeddings_seismic, corpus_ids = load_corpus_embeddings_parallel(
                    args.corpus_files, 
                    num_workers=args.num_workers
                )
            else:
                corpus_embeddings_seismic, corpus_ids = load_corpus_embeddings(
                    args.corpus_files
                )

            logger.info(f"Corpus loading time: {time.time() - start_time:.2f} seconds")
            
            if len(corpus_embeddings_seismic) == 0:
                raise ValueError("No corpus embeddings loaded!")
    else:
        # Still need corpus_ids for writing results, but don't need embeddings
        logger.info("Loading corpus IDs only (embeddings already in cached index)...")
        start_time = time.time()
        if len(args.corpus_files) > 1:
            _, corpus_ids = load_corpus_embeddings_parallel(
                args.corpus_files,
                num_workers=args.num_workers
            )
        else:
            _, corpus_ids = load_corpus_embeddings(
                args.corpus_files,
                num_workers=args.num_workers
            )
        logger.info(f"Corpus ID loading time: {time.time() - start_time:.2f} seconds")


    # Load query embeddings (always needed)
    start_time = time.time()
    query_embeddings_seismic, query_ids = load_query_embeddings(args.query_file)
    logger.info(f"Query loading time: {time.time() - start_time:.2f} seconds")
    
    if len(query_embeddings_seismic) == 0:
        raise ValueError("No query embeddings loaded!")
    
    # Perform retrieval
    logger.info("Performing retrieval with semantic_search_seismic_large_vocabulary...")
    
    start_time = time.time()
    results, search_time, corpus_index = semantic_search_seismic_large_vocabulary(
        query_embeddings_seismic,
        corpus_embeddings_decoded=corpus_embeddings_seismic,
        corpus_index=corpus_index,
        encodings_path=encodings_path,
        top_k=args.top_k,
        output_index=True,
    )
    
    total_time = time.time() - start_time
    logger.info(f"Retrieval complete in {total_time:.2f} seconds")
    logger.info(f"Search time (reported by seismic): {search_time:.6f} seconds")
    
    # Save index cache if requested
    if args.index_cache and corpus_index and not index_loaded:
        logger.info(f"Saving index to {args.index_cache}...")
        os.makedirs(os.path.dirname(args.index_cache), exist_ok=True)
        try:
            with open(args.index_cache, 'wb') as f:
                pickle.dump(corpus_index, f)
            logger.info("Index saved successfully")
        except Exception as e:
            logger.warning(f"Failed to save index: {e}")
    
    # Write results
    write_rank_file(results, query_ids, args.output_path)
    
    logger.info("=" * 60)
    logger.info("RETRIEVAL COMPLETE")
    logger.info(f"Output: {args.output_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
