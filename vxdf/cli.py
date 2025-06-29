"""Command-line utilities for working with VXDF files.

Usage examples:

# Show basic info about a VXDF file
python -m vxdf info sample.vxdf

# List document IDs in a VXDF file
python -m vxdf list sample.vxdf

# Extract a document by ID to stdout (pretty-printed JSON)
python -m vxdf get sample.vxdf doc_123 > doc.json

# Pack JSON lines into a VXDF file (expects each line to be a JSON object with id,text,vector)
python -m vxdf pack input.jsonl output.vxdf --embedding-dim 768 --compression zlib
"""

import argparse
import os
from typing import List, Optional

try:
    import argcomplete  # type: ignore
except ImportError:  # pragma: no cover
    argcomplete = None
import json
import sys

from .auth import get_openai_api_key, prompt_and_save_openai_key
from .reader import VXDFReader
from .writer import VXDFWriter

# --------------------------
# Banner shown on interactive terminals
# --------------------------
BANNER = r"""
__      __ __   __  _____ 
\ \    / / \ \ / / |  __ \
 \ \  / /   \ V /  | |  | |
  \ \/ /     > <   | |  | |
   \  /     / . \  | |__| |
    \/     /_/ \_\ |_____/
     Vector eXchange Data Format
     Store • Compress • Retrieve
"""

def _maybe_print_banner() -> None:
    """Print a friendly banner unless output is redirected or disabled.

    Suppressed when output is not a TTY or when the environment variable
    VXDF_NO_BANNER is set to any value.
    """
    if not sys.stdout.isatty():
        return
    if os.getenv("VXDF_NO_BANNER"):
        return
    print(BANNER)



def cmd_info(args: argparse.Namespace) -> None:
    with VXDFReader(args.file) as reader:
        print(json.dumps(reader.header, indent=2))
        print(f"\nTotal documents: {len(reader.offset_index)}")


def cmd_list(args: argparse.Namespace) -> None:
    with VXDFReader(args.file) as reader:
        for doc_id in reader.offset_index.keys():
            print(doc_id)


def cmd_get(args: argparse.Namespace) -> None:
    with VXDFReader(args.file) as reader:
        chunk = reader.get_chunk(args.doc_id)
        json.dump(chunk, sys.stdout, indent=2)
        sys.stdout.write("\n")


def cmd_pack(args: argparse.Namespace) -> None:
    """Pack newline-delimited JSON into a VXDF file."""
    emb_dim = args.embedding_dim
    compression = args.compression
    out_path = args.output

    writer = VXDFWriter(out_path, embedding_dim=emb_dim, compression=compression)
    count = 0
    istream = sys.stdin if args.input == "-" else open(args.input, encoding="utf-8")
    with istream as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            writer.add_chunk(data)
            count += 1
    writer.close()
    print(f"Wrote {count} documents to {out_path} (compression={compression})")


from .ingest import convert as ingest_convert
from .merge_split import merge as merge_vxdf
from .merge_split import split as split_vxdf
from .update import update as update_vxdf


def cmd_update(args: argparse.Namespace) -> None:
    # Resolve / prompt for API key first
    key = _resolve_openai_key(args.openai_key)
    update_vxdf(
        args.file,
        args.input,
        model=_resolve_model(args.model, key),
        dedupe=args.dedupe,
        openai_key=key,
        recursive=args.recursive,
        show_progress=args.progress,
    )

def cmd_merge(args: argparse.Namespace) -> None:
    merge_vxdf(
        args.output,
        args.inputs,
        dedupe=args.dedupe,
        show_progress=args.progress,
    )

def cmd_split(args: argparse.Namespace) -> None:
    split_vxdf(
        args.input,
        size_bytes=args.size,
        chunks_per_file=args.chunks,
        show_progress=args.progress,
    )

def _resolve_openai_key(cli_key: Optional[str]) -> Optional[str]:
    """Return an OpenAI key, prompting the user once if necessary."""
    if cli_key:
        return cli_key

    # env var or config already set?
    try:
        return get_openai_api_key()
    except Exception:
        # Not available – prompt the user (interactive only)
        return prompt_and_save_openai_key()


def _resolve_model(selected: str, openai_key: Optional[str] = None) -> str:
    """Return concrete model name based on *selected* flag and OpenAI key presence."""
    if selected in {"auto", ""}:
        if openai_key or os.getenv("OPENAI_API_KEY"):
            return "text-embedding-3-large"
        return "all-MiniLM-L6-v2"
    if selected == "openai":
        return "text-embedding-3-large"
    return selected


def cmd_convert(args: argparse.Namespace) -> None:
    # Resolve / prompt for API key first
    key = _resolve_openai_key(args.openai_key)

    ingest_convert(
        args.input,
        args.output,
        model=_resolve_model(args.model, key),
        compression=args.compression,
        openai_key=key,
        recursive=args.recursive,
        resume=args.resume,
        workers=args.workers,
        show_progress=args.progress,
        detect_pii=args.detect_pii,
        pii_patterns=args.pii_patterns,
    )
    print(f"Converted {args.input} -> {args.output}")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vxdf",
        description="VXDF command-line utilities",
        epilog=(
            "Authentication: OpenAI API key is resolved in order from --openai-key, "
            "the OPENAI_API_KEY environment variable, or ~/.vxdf/config.toml. "
            "AWS and GCP access use the standard boto3 / google-auth credential chains."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Propagate exceptions instead of friendly messages (debugging).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # info
    p_info = subparsers.add_parser("info", help="Show header and stats of a VXDF file")
    p_info.add_argument("file", help="Path to .vxdf file")
    p_info.set_defaults(func=cmd_info)

    # list
    p_list = subparsers.add_parser("list", help="List document IDs in a VXDF file")
    p_list.add_argument("file", help="Path to .vxdf file")
    p_list.set_defaults(func=cmd_list)

    # get
    p_get = subparsers.add_parser("get", help="Extract a document by ID from a VXDF file")
    p_get.add_argument("file", help="Path to .vxdf file")
    p_get.add_argument("doc_id", help="Document ID to retrieve")
    p_get.set_defaults(func=cmd_get)

    # update
    p_up = subparsers.add_parser("update", help="Append new data to an existing VXDF")
    p_up.add_argument("file", help="Existing .vxdf file to update")
    p_up.add_argument("input", help="Input file/folder with new data")
    p_up.add_argument(
        "--model",
        default="auto",
        help=(
            "Embedding model to use when updating. Same semantics as --model for convert: 'auto' (default) picks OpenAI text-embedding-3-large "
            "when key is present else all-MiniLM-L6-v2)."
        ),
    )
    p_up.add_argument("--dedupe", choices=["skip", "overwrite", "error"], default="skip", help="Duplicate ID handling")
    p_up.add_argument("--no-progress", dest="progress", action="store_false", help="Disable progress bars")
    p_up.add_argument("--openai-key", dest="openai_key", help="OpenAI API key (overrides env var / config file)")
    p_up.add_argument("-r", "--recursive", action="store_true", help="Recurse into directories when INPUT is folder")
    p_up.set_defaults(func=cmd_update, progress=True)

    # merge
    p_merge = subparsers.add_parser("merge", help="Merge multiple VXDF files into one")
    p_merge.add_argument("output", help="Output .vxdf file")
    p_merge.add_argument("inputs", nargs="+", help="Input VXDF files to merge")
    p_merge.add_argument("--dedupe", choices=["skip", "error", "firstwins"], default="skip", help="Duplicate ID handling")
    p_merge.add_argument("--no-progress", dest="progress", action="store_false", help="Disable progress bars")
    p_merge.set_defaults(func=cmd_merge, progress=True)

    # split
    p_split = subparsers.add_parser("split", help="Split a large VXDF into parts")
    p_split.add_argument("input", help="Input .vxdf file to split")
    p_split.add_argument("--size", type=int, help="Target size per part in bytes (mutually exclusive with --chunks)")
    p_split.add_argument("--chunks", type=int, help="Max chunks per part (mutually exclusive with --size)")
    p_split.add_argument("--no-progress", dest="progress", action="store_false", help="Disable progress bars")
    p_split.set_defaults(func=cmd_split, progress=True)

    # convert
    p_conv = subparsers.add_parser("convert", help="Convert common formats (PDF, CSV, etc.) to VXDF")
    p_conv.add_argument("input", help="Input file (pdf, csv, xlsx, docx, json, parquet, txt, md ...)")
    p_conv.add_argument("output", help="Output .vxdf file")
    p_conv.add_argument(
        "--model",
        default="auto",
        help=(
            "Embedding model to use. Choices: 'auto' (default, picks OpenAI text-embedding-3-large when key is present "
            "else all-MiniLM-L6-v2), 'openai' (force text-embedding-3-large), or any Sentence-Transformers/HF model name."
        ),
    )
    p_conv.add_argument("--compression", choices=["none", "zlib", "zstd"], default="none", help="Chunk compression")
    p_conv.add_argument("--no-progress", dest="progress", action="store_false", help="Disable progress bars")
    p_conv.add_argument("--resume", action="store_true", help="Resume an interrupted conversion job")
    p_conv.add_argument("--workers", type=int, default=1, help="Number of parallel embedding workers")
    p_conv.add_argument(
        "--openai-key",
        dest="openai_key",
        help="OpenAI API key (if omitted, OPENAI_API_KEY env var or ~/.vxdf/config.toml is used).",
    )
    p_conv.add_argument("-r", "--recursive", action="store_true", help="Recurse into subdirectories when INPUT is a folder")
    p_conv.add_argument("--no-pii", dest="detect_pii", action="store_false", help="Disable automatic PII detection")
    p_conv.add_argument("--pii-pattern", dest="pii_patterns", action="append", help="Custom regex pattern(s) to flag as sensitive; can be repeated")
    p_conv.set_defaults(func=cmd_convert, progress=True, detect_pii=True, pii_patterns=None)

    # pack
    p_pack = subparsers.add_parser("pack", help="Pack newline-delimited JSON into a VXDF file")
    p_pack.add_argument("input", help="Input JSONL file (each line a JSON object)")
    p_pack.add_argument("output", help="Output .vxdf file")
    p_pack.add_argument("--embedding-dim", type=int, default=768, dest="embedding_dim", help="Embedding dimension (default 768)")
    p_pack.add_argument("--compression", choices=["none", "zlib", "zstd"], default="none", help="Chunk compression")
    p_pack.set_defaults(func=cmd_pack)

    return parser


from . import errors as vxdf_errors


def _run_command(func, parsed_args):
    """Run command with unified error handling."""
    try:
        func(parsed_args)
    except vxdf_errors.MissingDependencyError as exc:
        print(f"\u001b[31mMissing dependency:\u001b[0m {exc}", file=sys.stderr)
        print("Tip: install optional extras or the suggested pip package and retry.", file=sys.stderr)
        sys.exit(1)
    except vxdf_errors.AuthenticationError as exc:
        print(f"\u001b[31mAuth error:\u001b[0m {exc}", file=sys.stderr)
        print("Tip: configure credentials via env vars, config file, or CLI flags.", file=sys.stderr)
        sys.exit(1)
    except vxdf_errors.NetworkError as exc:
        print(f"\u001b[31mNetwork error:\u001b[0m {exc}", file=sys.stderr)
        print("Tip: check your internet connection or proxy settings and retry.", file=sys.stderr)
        sys.exit(1)
    except vxdf_errors.VXDFError as exc:
        print(f"\u001b[31mError:\u001b[0m {exc}", file=sys.stderr)
        if parsed_args.strict:
            raise
        sys.exit(1)

def main(argv: Optional[List[str]] = None) -> None:
    _maybe_print_banner()
    parser = build_parser()
    if argcomplete is not None:
        argcomplete.autocomplete(parser)
    args = parser.parse_args(argv)
    _run_command(args.func, args)


if __name__ == "__main__":
    main()
