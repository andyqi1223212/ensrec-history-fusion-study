"""Build and verify the frozen SentenceT5 item table used by the Text route."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from tfrecord.reader import tfrecord_loader

MODEL_ID = "sentence-transformers/sentence-t5-xxl"
MODEL_DIMENSION = 768
PLACEHOLDERS = 2


def text_value(value) -> str:
    """Extract one TFRecord text scalar without silently stringifying arrays."""
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"Expected one text scalar, got shape {array.shape}")
    scalar = array.reshape(-1)[0]
    if isinstance(scalar, bytes):
        return scalar.decode("utf-8")
    if isinstance(scalar, np.bytes_):
        return bytes(scalar).decode("utf-8")
    if isinstance(scalar, str):
        return scalar
    raise TypeError(f"Expected UTF-8 text, got {type(scalar).__name__}")


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def records(data_dir: Path, split: str):
    paths = sorted((data_dir / split).glob("*.tfrecord.gz"))
    if not paths:
        raise FileNotFoundError(f"No TFRecord files in {data_dir / split}")
    for path in paths:
        yield from tfrecord_loader(str(path), None, compression_type="gzip")


def read_catalog(data_dir: Path) -> list[tuple[int, str]]:
    catalog = []
    for row in records(data_dir, "items"):
        ids = np.asarray(row["id"]).reshape(-1)
        if ids.size != 1:
            raise ValueError(f"Expected one metadata ID, got shape {ids.shape}")
        catalog.append((int(ids[0]), text_value(row["text"])))
    if not catalog:
        raise ValueError("Item catalog is empty; no embeddings can be built")
    ids = [item_id for item_id, _ in catalog]
    expected = list(range(1, len(catalog) + 1))
    if sorted(ids) != expected:
        raise ValueError("Metadata IDs must uniquely and completely cover 1..N")
    return sorted(catalog, key=lambda pair: pair[0])


def fake_encode(texts: list[str], dimension: int) -> torch.Tensor:
    """Stable, download-free test encoder; never suitable for model results."""
    rows = []
    for text in texts:
        seed = hashlib.sha256(text.encode("utf-8")).digest()
        raw = (seed * math.ceil(dimension / len(seed)))[:dimension]
        rows.append([(byte - 127.5) / 127.5 for byte in raw])
    return torch.tensor(rows, dtype=torch.float32)


def sentence_t5_encoder(device: str | None = None) -> Callable[[list[str]], torch.Tensor]:
    # Import lazily: fake mode and validation do not install or load model weights.
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_ID, device=device)

    def encode(texts: list[str]) -> torch.Tensor:
        vectors = model.encode(texts, convert_to_tensor=True, show_progress_bar=False)
        return vectors.detach().to(device="cpu", dtype=torch.float32)

    return encode


def encode_catalog(
    catalog: list[tuple[int, str]],
    encoder: Callable[[list[str]], torch.Tensor],
    dimension: int,
    batch_size: int,
) -> tuple[torch.Tensor, list[dict]]:
    if batch_size < 1 or dimension < 1:
        raise ValueError("batch_size and dimension must be positive")
    if not catalog:
        raise ValueError("Item catalog is empty; no embeddings can be built")
    catalog = sorted(catalog, key=lambda pair: pair[0])
    ids = [item_id for item_id, _ in catalog]
    if ids != list(range(1, len(catalog) + 1)):
        raise ValueError("Catalog IDs must uniquely and completely cover 1..N")

    encoded_batches = []
    for start in range(0, len(catalog), batch_size):
        texts = [text for _, text in catalog[start : start + batch_size]]
        batch = torch.as_tensor(encoder(texts)).detach().to(device="cpu", dtype=torch.float32)
        if batch.shape != (len(texts), dimension):
            raise ValueError(
                f"Encoder returned {tuple(batch.shape)}, expected {(len(texts), dimension)}"
            )
        encoded_batches.append(batch)
    real_vectors = torch.cat(encoded_batches, dim=0)
    table = torch.cat(
        [torch.zeros((PLACEHOLDERS, dimension), dtype=torch.float32), real_vectors], dim=0
    )
    manifest_rows = [
        {
            "row": item_id + 1,
            "metadata_id": item_id,
            "source_text_hash": text_hash(text),
        }
        for item_id, text in catalog
    ]
    return table, manifest_rows


def verify_table(table: torch.Tensor, manifest_rows: list[dict], item_count: int, dimension: int):
    if not isinstance(table, torch.Tensor):
        raise TypeError("Embedding table must be a torch.Tensor")
    expected_shape = (item_count + PLACEHOLDERS, dimension)
    if tuple(table.shape) != expected_shape:
        raise ValueError(f"Table shape is {tuple(table.shape)}, expected {expected_shape}")
    if not torch.is_floating_point(table) or not torch.isfinite(table).all():
        raise ValueError("Embedding table must contain only finite floating-point values")
    if not torch.equal(table[:PLACEHOLDERS], torch.zeros_like(table[:PLACEHOLDERS])):
        raise ValueError("Rows 0 and 1 must be exact zero placeholder vectors")
    expected_ids = list(range(1, item_count + 1))
    observed_ids = [entry.get("metadata_id") for entry in manifest_rows]
    expected_rows = [item_id + 1 for item_id in expected_ids]
    observed_rows = [entry.get("row") for entry in manifest_rows]
    if observed_ids != expected_ids or observed_rows != expected_rows:
        raise ValueError("Manifest must cover IDs 1..N once in order at row=metadata_id+1")
    if any(
        not isinstance(entry.get("source_text_hash"), str)
        or len(entry["source_text_hash"]) != 64
        for entry in manifest_rows
    ):
        raise ValueError("Each manifest row must contain a SHA-256 source_text_hash")


def tensor_sha256(table: torch.Tensor) -> str:
    """Hash dtype, shape, and canonical contiguous CPU tensor bytes."""
    tensor = table.detach().to(device="cpu").contiguous()
    header = json.dumps(
        {"dtype": str(tensor.dtype), "shape": list(tensor.shape)}, sort_keys=True
    ).encode("utf-8")
    payload = tensor.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(header + b"\0" + payload).hexdigest()


def verify_tensor_sha256(table: torch.Tensor, manifest: dict):
    expected = manifest.get("tensor_sha256")
    if not isinstance(expected, str) or tensor_sha256(table) != expected:
        raise ValueError("Tensor SHA-256 mismatch; the saved table differs from its manifest")


def verify_embedding_artifact(table: torch.Tensor, manifest: dict):
    item_count = manifest["item_count"]
    dimension = manifest["dimension"]
    if manifest.get("encoder_model") == MODEL_ID and dimension != MODEL_DIMENSION:
        raise ValueError(f"{MODEL_ID} manifest dimension must be {MODEL_DIMENSION}")
    verify_table(table, manifest["rows"], item_count, dimension)
    verify_tensor_sha256(table, manifest)


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_event_text_hashes(data_dir: Path, manifest: dict) -> dict:
    by_metadata_id = {
        entry["metadata_id"]: entry["source_text_hash"] for entry in manifest["rows"]
    }
    reports = {}
    for split in ("training", "evaluation", "testing"):
        identity_hashes: dict[int, str] = {}
        events = 0
        for row in records(data_dir, split):
            sequence = np.asarray(row["sequence_data"]).reshape(-1)
            if sequence.size == 0:
                raise ValueError(f"{split}: empty sequence_data")
            sequence_id = int(sequence[0])
            metadata_id = sequence_id + 1
            event_hash = text_hash(text_value(row["text"]))
            expected_hash = by_metadata_id.get(metadata_id)
            if expected_hash is None or event_hash != expected_hash:
                raise ValueError(
                    f"{split}: first-item text hash mismatch for sequence ID {sequence_id} "
                    f"(metadata ID {metadata_id})"
                )
            previous = identity_hashes.setdefault(sequence_id, event_hash)
            if previous != event_hash:
                raise ValueError(f"{split}: inconsistent event text for sequence ID {sequence_id}")
            events += 1
        reports[split] = {
            "events_verified": events,
            "unique_first_item_ids_verified": len(identity_hashes),
        }
    counts = {report["unique_first_item_ids_verified"] for report in reports.values()}
    if len(counts) != 1:
        raise ValueError("Expected the same unique first-item identity count in every split")
    return reports


def verify_catalog_hashes(data_dir: Path, manifest: dict) -> int:
    expected = {
        entry["metadata_id"]: entry["source_text_hash"] for entry in manifest["rows"]
    }
    catalog = read_catalog(data_dir)
    for metadata_id, text in catalog:
        if expected.get(metadata_id) != text_hash(text):
            raise ValueError(f"Catalog text hash mismatch for metadata ID {metadata_id}")
    if len(expected) != len(catalog):
        raise ValueError("Manifest and current item catalog have different item counts")
    return len(catalog)


def write_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def build(args):
    catalog = read_catalog(args.data_dir)
    if args.encoder == "fake":
        dimension = args.dimension or 8
        encoder = lambda texts: fake_encode(texts, dimension)
        model_id = "fake-sha256-test-encoder"
    else:
        dimension = args.dimension or MODEL_DIMENSION
        if dimension != MODEL_DIMENSION:
            raise ValueError(f"{MODEL_ID} must use dimension {MODEL_DIMENSION}")
        encoder = sentence_t5_encoder(args.device)
        model_id = MODEL_ID
    table, rows = encode_catalog(catalog, encoder, dimension, args.batch_size)
    verify_table(table, rows, len(catalog), dimension)
    manifest = {
        "format_version": 1,
        "encoder_model": model_id,
        "dimension": dimension,
        "item_count": len(catalog),
        "tensor_sha256": tensor_sha256(table),
        "tensor_sha256_covers": "canonical CPU tensor bytes, dtype and shape",
        "source_order_contract": (
            "Input items TFRecord has metadata IDs 1..N exactly once; rows are sorted by ID; "
            "real item metadata_id m is stored at tensor row m+1; rows 0 and 1 are zeros."
        ),
        "rows": rows,
    }
    catalog_items_verified = verify_catalog_hashes(args.data_dir, manifest)
    manifest["event_text_hash_audit"] = verify_event_text_hashes(args.data_dir, manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(table, temporary)
    temporary.replace(args.output)
    write_json(args.manifest, manifest)
    print(
        json.dumps(
            {
                "embedding_path": str(args.output),
                "manifest_path": str(args.manifest),
                "shape": list(table.shape),
                "encoder_model": model_id,
                "catalog_items_verified": catalog_items_verified,
                "event_text_hash_audit": manifest["event_text_hash_audit"],
            },
            indent=2,
        )
    )


def validate(args):
    manifest = load_manifest(args.manifest)
    table = torch.load(args.embeddings, map_location="cpu", weights_only=True)
    verify_embedding_artifact(table, manifest)
    catalog_items_verified = verify_catalog_hashes(args.data_dir, manifest)
    reports = verify_event_text_hashes(args.data_dir, manifest)
    print(
        json.dumps(
            {
                "valid": True,
                "shape": list(table.shape),
                "catalog_items_verified": catalog_items_verified,
                "event_text_hash_audit": reports,
            },
            indent=2,
        )
    )


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build", help="encode item texts and write a table plus manifest")
    build_parser.add_argument("--data-dir", type=Path, required=True)
    build_parser.add_argument("--output", type=Path, required=True)
    build_parser.add_argument("--manifest", type=Path, required=True)
    build_parser.add_argument(
        "--encoder", choices=("fake", "sentence-t5-xxl"), default="sentence-t5-xxl"
    )
    build_parser.add_argument("--dimension", type=int)
    build_parser.add_argument("--batch-size", type=int, default=8)
    build_parser.add_argument("--device")
    build_parser.set_defaults(func=build)
    validate_parser = sub.add_parser("validate", help="verify tensor, manifest and Beauty event hashes")
    validate_parser.add_argument("--data-dir", type=Path, required=True)
    validate_parser.add_argument("--embeddings", type=Path, required=True)
    validate_parser.add_argument("--manifest", type=Path, required=True)
    validate_parser.set_defaults(func=validate)
    return root


def main():
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
