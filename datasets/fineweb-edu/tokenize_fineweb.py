#!/usr/bin/env python3
"""
FineWeb-Edu Tokenization Script

Converts FineWeb-Edu dataset to tokenized shards for language model training.
Uses streaming mode by default for memory efficiency. Supports resumable processing
and flexible tokenizers.

Usage:
    python tokenize_fineweb.py --help
    python tokenize_fineweb.py --subset sample-10BT --output-dir ./shards
    python tokenize_fineweb.py --parallel  # Use parallel mode for faster processing
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import tiktoken
from datasets import load_dataset
from transformers import AutoTokenizer

# Approximate token counts for known subsets (for ETA estimation in streaming mode)
SUBSET_TOKEN_ESTIMATES = {
    "sample-10BT": 10_000_000_000,
    "sample-100BT": 100_000_000_000,
}


def format_eta(seconds: float) -> str:
    """Format seconds into human-readable time string."""
    if seconds < 0 or not np.isfinite(seconds):
        return "unknown"
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes}m"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def get_tokenizer(tokenizer_name: str) -> tuple:
    """
    Load tokenizer by name.

    Args:
        tokenizer_name: Either 'gpt2' for tiktoken or a HuggingFace model name

    Returns:
        Tuple of (encode_fn, eot_token, vocab_size)
    """
    if tokenizer_name == "gpt2":
        enc = tiktoken.get_encoding("gpt2")
        eot = enc._special_tokens["<|endoftext|>"]
        return enc.encode_ordinary, eot, 50257
    else:
        enc = AutoTokenizer.from_pretrained(tokenizer_name)
        eot = enc.eos_token_id
        return lambda text: enc.encode(text, add_special_tokens=False), eot, len(enc)


def tokenize_doc(text: str, encode_fn, eot: int) -> list[int]:
    """Tokenize a single document with EOT prefix."""
    return [eot] + encode_fn(text)


def write_shard(shard: np.ndarray, shard_idx: int, output_dir: Path, val_every: int):
    """Write a shard to disk."""
    split = "valid" if shard_idx % val_every == 0 else "train"
    path = output_dir / f"fineweb_edu_{split}_{shard_idx:05d}.npz"
    np.savez(path, tokens=shard)
    return path


def load_checkpoint(checkpoint_path: Path) -> dict | None:
    """Load checkpoint if it exists."""
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            return json.load(f)
    return None


def save_checkpoint(checkpoint_path: Path, data: dict):
    """Save checkpoint to disk."""
    with open(checkpoint_path, "w") as f:
        json.dump(data, f)


def process_streaming(args):
    """Process dataset in streaming mode (memory efficient)."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.json"

    # Get estimated total tokens for ETA calculation
    estimated_total_tokens = SUBSET_TOKEN_ESTIMATES.get(args.subset)

    # Load tokenizer
    print(f"Loading tokenizer: {args.tokenizer}")
    encode_fn, eot, vocab_size = get_tokenizer(args.tokenizer)
    print(f"Vocabulary size: {vocab_size:,}")

    # Check for checkpoint
    checkpoint = load_checkpoint(checkpoint_path)
    start_doc = 0
    start_shard = 0
    if checkpoint and not args.restart:
        start_doc = checkpoint["doc_offset"]
        start_shard = checkpoint["shard_idx"]
        print(f"Resuming from document {start_doc:,}, shard {start_shard}")

    # Load dataset in streaming mode
    print(f"Loading dataset: {args.dataset} ({args.subset})")
    dataset = load_dataset(
        args.dataset,
        args.subset,
        split="train",
        streaming=True,
    )

    # Skip to checkpoint position
    if start_doc > 0:
        print(f"Skipping to document {start_doc:,}...")
        dataset = dataset.skip(start_doc)

    # Initialize shard
    shard = np.empty((args.shard_size,), dtype=np.uint16)
    shard_token_count = 0
    shard_idx = start_shard

    docs_processed = start_doc
    tokens_generated = 0
    shards_written = 0
    start_time = time.time()
    last_checkpoint_time = start_time

    print("Processing documents...")

    for doc in dataset:
        tokens = tokenize_doc(doc["text"], encode_fn, eot)
        docs_processed += 1
        tokens_generated += len(tokens)

        # Progress update
        if docs_processed % 10000 == 0:
            elapsed = time.time() - start_time
            docs_per_sec = (docs_processed - start_doc) / elapsed if elapsed > 0 else 0
            tokens_per_sec = tokens_generated / elapsed if elapsed > 0 else 0

            # Calculate ETA if we have an estimate
            if estimated_total_tokens and tokens_per_sec > 0:
                remaining_tokens = estimated_total_tokens - tokens_generated
                eta_seconds = remaining_tokens / tokens_per_sec
                eta_str = f" | ETA: {format_eta(eta_seconds)}"
                pct = tokens_generated / estimated_total_tokens * 100
                pct_str = f" ({pct:.1f}%)"
            else:
                eta_str = ""
                pct_str = ""

            print(
                f"Docs: {docs_processed:,} | "
                f"Tokens: {tokens_generated:,}{pct_str} | "
                f"Shards: {shards_written} | "
                f"Speed: {docs_per_sec:.0f} docs/s{eta_str}",
                end="\r",
            )

        # Fill shard
        while len(tokens) > 0:
            space_left = args.shard_size - shard_token_count

            if len(tokens) <= space_left:
                # All tokens fit in current shard
                shard[shard_token_count : shard_token_count + len(tokens)] = tokens
                shard_token_count += len(tokens)
                tokens = []
            else:
                # Fill current shard and write it
                shard[shard_token_count:] = tokens[:space_left]
                write_shard(shard, shard_idx, output_dir, args.val_every)
                shards_written += 1
                shard_idx += 1

                # Start new shard with remaining tokens
                tokens = tokens[space_left:]
                shard_token_count = 0

                # Save checkpoint periodically (every 5 minutes)
                if time.time() - last_checkpoint_time > 300:
                    save_checkpoint(checkpoint_path, {
                        "doc_offset": docs_processed,
                        "shard_idx": shard_idx,
                        "tokens_generated": tokens_generated,
                    })
                    last_checkpoint_time = time.time()

    # Write final shard (may be partial)
    if shard_token_count > 0:
        # Trim to actual size
        final_shard = shard[:shard_token_count]
        write_shard(final_shard, shard_idx, output_dir, args.val_every)
        shards_written += 1

    # Remove checkpoint on completion
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    elapsed = time.time() - start_time
    print()
    print(f"Completed in {elapsed:.2f} seconds")
    print(f"Total documents: {docs_processed:,}")
    print(f"Total tokens: {tokens_generated:,}")
    print(f"Total shards: {shards_written}")


def process_parallel(args):
    """Process dataset with parallel workers (faster but uses more memory)."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.json"

    # Load tokenizer
    print(f"Loading tokenizer: {args.tokenizer}")
    encode_fn, eot, vocab_size = get_tokenizer(args.tokenizer)
    print(f"Vocabulary size: {vocab_size:,}")

    # Check for checkpoint
    checkpoint = load_checkpoint(checkpoint_path)
    start_doc = 0
    start_shard = 0
    if checkpoint and not args.restart:
        start_doc = checkpoint["doc_offset"]
        start_shard = checkpoint["shard_idx"]
        print(f"Resuming from document {start_doc:,}, shard {start_shard}")

    # Load full dataset
    print(f"Loading dataset: {args.dataset} ({args.subset})")
    print("Note: This loads the full dataset into memory. Use --streaming for lower memory usage.")
    dataset = load_dataset(
        args.dataset,
        args.subset,
        split="train",
        cache_dir=args.cache_dir,
    )

    total_docs = len(dataset)
    print(f"Total documents: {total_docs:,}")

    num_workers = args.num_workers or os.cpu_count()
    print(f"Using {num_workers} workers")

    # Tokenization function for parallel processing
    def tokenize_idx(idx: int) -> list[int]:
        return tokenize_doc(dataset[idx]["text"], encode_fn, eot)

    # Initialize shard
    shard = np.empty((args.shard_size,), dtype=np.uint16)
    shard_token_count = 0
    shard_idx = start_shard

    docs_processed = start_doc
    tokens_generated = 0
    shards_written = 0
    start_time = time.time()
    last_checkpoint_time = start_time

    chunksize = max(1, (total_docs - start_doc) // (num_workers * 100))

    print("Processing documents...")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for tokens in executor.map(tokenize_idx, range(start_doc, total_docs), chunksize=chunksize):
            docs_processed += 1
            tokens_generated += len(tokens)

            # Progress update
            if docs_processed % 10000 == 0:
                elapsed = time.time() - start_time
                docs_per_sec = (docs_processed - start_doc) / elapsed if elapsed > 0 else 0
                pct = docs_processed / total_docs * 100

                # Calculate ETA based on remaining docs
                if docs_per_sec > 0:
                    remaining_docs = total_docs - docs_processed
                    eta_seconds = remaining_docs / docs_per_sec
                    eta_str = f" | ETA: {format_eta(eta_seconds)}"
                else:
                    eta_str = ""

                print(
                    f"Docs: {docs_processed:,}/{total_docs:,} ({pct:.1f}%) | "
                    f"Tokens: {tokens_generated:,} | "
                    f"Shards: {shards_written} | "
                    f"Speed: {docs_per_sec:.0f} docs/s{eta_str}",
                    end="\r",
                )

            # Fill shard
            while len(tokens) > 0:
                space_left = args.shard_size - shard_token_count

                if len(tokens) <= space_left:
                    shard[shard_token_count : shard_token_count + len(tokens)] = tokens
                    shard_token_count += len(tokens)
                    tokens = []
                else:
                    shard[shard_token_count:] = tokens[:space_left]
                    write_shard(shard, shard_idx, output_dir, args.val_every)
                    shards_written += 1
                    shard_idx += 1

                    tokens = tokens[space_left:]
                    shard_token_count = 0

                    # Save checkpoint periodically
                    if time.time() - last_checkpoint_time > 300:
                        save_checkpoint(checkpoint_path, {
                            "doc_offset": docs_processed,
                            "shard_idx": shard_idx,
                            "tokens_generated": tokens_generated,
                        })
                        last_checkpoint_time = time.time()

    # Write final shard
    if shard_token_count > 0:
        final_shard = shard[:shard_token_count]
        write_shard(final_shard, shard_idx, output_dir, args.val_every)
        shards_written += 1

    # Remove checkpoint on completion
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    elapsed = time.time() - start_time
    print()
    print(f"Completed in {elapsed:.2f} seconds")
    print(f"Total documents: {docs_processed:,}")
    print(f"Total tokens: {tokens_generated:,}")
    print(f"Total shards: {shards_written}")


def main():
    parser = argparse.ArgumentParser(
        description="Tokenize FineWeb-Edu dataset for language model training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset options
    parser.add_argument(
        "--dataset",
        type=str,
        default="HuggingFaceFW/fineweb-edu",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="sample-100BT",
        choices=["sample-10BT", "sample-100BT", "default"],
        help="Dataset subset to use",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./cache",
        help="Directory to cache downloaded dataset (parallel mode only)",
    )

    # Tokenizer options
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="gpt2",
        help="Tokenizer to use: 'gpt2' for tiktoken or HuggingFace model name",
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./processed",
        help="Directory to write tokenized shards",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=100_000_000,
        help="Number of tokens per shard",
    )
    parser.add_argument(
        "--val-every",
        type=int,
        default=100,
        help="Every Nth shard goes to validation split",
    )

    # Processing options
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Use parallel mode (faster but loads full dataset into memory)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count, parallel mode only)",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Ignore checkpoint and restart from beginning",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("FineWeb-Edu Tokenization")
    print("=" * 60)
    print(f"Dataset: {args.dataset} ({args.subset})")
    print(f"Tokenizer: {args.tokenizer}")
    print(f"Output: {args.output_dir}")
    print(f"Shard size: {args.shard_size:,} tokens")
    print(f"Mode: {'parallel' if args.parallel else 'streaming'}")
    print("=" * 60)

    if args.parallel:
        process_parallel(args)
    else:
        process_streaming(args)


if __name__ == "__main__":
    main()
