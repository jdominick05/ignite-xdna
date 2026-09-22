# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
src/ignite_xdna/compiler/serializer.py

Monolithic Binary Container Specification (.ignite) for AMD XDNA1 / Phoenix NPU.
Packages multi-stage transaction sequences, stationary weights, quantization scales,
and graph metadata into a single deployable artifact with 64-byte aligned zero-copy
memory-mapped sections.

Container Format Layout:
  ┌────────────────────────────────────────────────────────┐
  │ Header (64 bytes, fixed size):                         │
  │   - Magic: b"IGNT" (4 bytes)                           │
  │   - Version: uint16 (1)                                │
  │   - Arch ID: uint16 (1 = XDNA1/Phoenix)                │
  │   - Header Size: uint32 (64)                           │
  │   - CRC32: uint32 (checksum of file from offset 64)    │
  │   - Total File Size: uint64                            │
  │   - Manifest Offset: uint64 (64)                       │
  │   - Manifest Size: uint64 (unpadded JSON bytes)        │
  │   - Blob Section Offset: uint64 (64-byte aligned)      │
  │   - Blob Section Size: uint64                          │
  │   - Num Blobs: uint32                                  │
  │   - Reserved: 4 bytes                                  │
  ├────────────────────────────────────────────────────────┤
  │ Manifest Section (64-byte aligned):                    │
  │   - UTF-8 JSON describing layer dimensions, per-tensor │
  │     quantization scales, I/O shapes, anchor strides,   │
  │     stage definitions, and the blob directory.         │
  ├────────────────────────────────────────────────────────┤
  │ Blob Section (each blob strictly 64-byte aligned):     │
  │   - Packed transaction binaries (init.bin, exec.bin)   │
  │   - Pre-swizzled weight and bias arrays                │
  └────────────────────────────────────────────────────────┘
"""

import hashlib
import json
import mmap
import os
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


MAGIC_BYTES = b"IGNT"
FORMAT_VERSION = 1
ARCH_XDNA1_PHOENIX = 1
ARCH_XDNA2_STRIX = 2

# Graph-engine identities. They live here because the serializer is the one module the compiler
# and the runtime both already import. Before this they were two unlinked string literals, one
# in engine_compile.py and one in graph_session.py, with nothing keeping them in step.
ENGINE_CONV_INT8 = "conv_engine_v1"
ENGINE_CONV_BF16 = "conv_engine_bf16_v1"
GRAPH_ENGINES = (ENGINE_CONV_INT8, ENGINE_CONV_BF16)

# Activation element semantics, carried per placement as "elem". ABSENT MEANS ELEM_INT, so every
# container built before the bf16 engine existed keeps working without being rebuilt. The storage
# dtype stays a real numpy spelling ("uint8", "uint16") because np.dtype("bf16") raises outright
# and np.dtype("bfloat16") resolves only when ml_dtypes happens to have been imported first -
# an import-order dependent failure is worse than either alternative.
ELEM_INT = "int"
ELEM_BF16 = "bf16"

HEADER_SIZE = 64
HEADER_STRUCT_FORMAT = "<4s2H2I5QI4s"
ALIGNMENT = 64


def align_up(offset: int, alignment: int = ALIGNMENT) -> int:
    """Rounds offset up to the nearest multiple of alignment."""
    remainder = offset % alignment
    if remainder == 0:
        return offset
    return offset + (alignment - remainder)


@dataclass
class IgniteHeader:
    """Fixed-size 64-byte container header."""
    magic: bytes = MAGIC_BYTES
    version: int = FORMAT_VERSION
    arch_id: int = ARCH_XDNA1_PHOENIX
    header_size: int = HEADER_SIZE
    crc32: int = 0
    total_file_size: int = 0
    manifest_offset: int = HEADER_SIZE
    manifest_size: int = 0
    blob_offset: int = 0
    blob_size: int = 0
    num_blobs: int = 0
    reserved: bytes = b"\x00" * 4

    def pack(self) -> bytes:
        return struct.pack(
            HEADER_STRUCT_FORMAT,
            self.magic,
            self.version,
            self.arch_id,
            self.header_size,
            self.crc32,
            self.total_file_size,
            self.manifest_offset,
            self.manifest_size,
            self.blob_offset,
            self.blob_size,
            self.num_blobs,
            self.reserved,
        )

    @classmethod
    def unpack(cls, data: bytes) -> "IgniteHeader":
        if len(data) < HEADER_SIZE:
            raise ValueError(f"Header data too short: expected {HEADER_SIZE} bytes, got {len(data)}")
        (
            magic,
            version,
            arch_id,
            header_size,
            crc32,
            total_file_size,
            manifest_offset,
            manifest_size,
            blob_offset,
            blob_size,
            num_blobs,
            reserved,
        ) = struct.unpack(HEADER_STRUCT_FORMAT, data[:HEADER_SIZE])

        if magic != MAGIC_BYTES:
            raise ValueError(f"Invalid .ignite magic bytes: {magic!r} (expected {MAGIC_BYTES!r})")
        if header_size != HEADER_SIZE:
            raise ValueError(f"Invalid header size: {header_size} (expected {HEADER_SIZE})")
        if version != FORMAT_VERSION:
            raise ValueError(f"Unsupported .ignite format version {version} (expected {FORMAT_VERSION})")
        if arch_id not in (ARCH_XDNA1_PHOENIX, ARCH_XDNA2_STRIX):
            raise ValueError(f"Unknown .ignite arch id {arch_id}")
        if manifest_offset != HEADER_SIZE:
            raise ValueError(f"Manifest must follow the header at offset {HEADER_SIZE}, got {manifest_offset}")

        return cls(
            magic=magic,
            version=version,
            arch_id=arch_id,
            header_size=header_size,
            crc32=crc32,
            total_file_size=total_file_size,
            manifest_offset=manifest_offset,
            manifest_size=manifest_size,
            blob_offset=blob_offset,
            blob_size=blob_size,
            num_blobs=num_blobs,
            reserved=reserved,
        )


@dataclass
class IgniteBlobEntry:
    """Directory entry for a single 64-byte aligned blob."""
    name: str
    offset: int
    size: int
    content_type: str = "raw"
    crc32: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "offset": self.offset,
            "size": self.size,
            "content_type": self.content_type,
            "crc32": self.crc32,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "IgniteBlobEntry":
        return cls(
            name=d["name"],
            offset=int(d["offset"]),
            size=int(d["size"]),
            content_type=d.get("content_type", "raw"),
            crc32=int(d.get("crc32", 0)),
        )


class IgniteModelWriter:
    """
    Serializes multi-stage transaction bundles, weights, and model metadata
    into a zero-copy .ignite binary container.
    """

    def __init__(
        self,
        manifest_meta: Optional[Dict[str, Any]] = None,
        arch_id: int = ARCH_XDNA1_PHOENIX,
    ):
        self.arch_id = arch_id
        self.manifest = dict(manifest_meta or {})
        self._blobs: List[Tuple[str, bytes, str]] = []

    def add_blob(self, name: str, data: bytes, content_type: str = "raw") -> "IgniteModelWriter":
        """Adds a binary blob to be packed into the container."""
        if not isinstance(name, str) or not name:
            raise ValueError("Blob name must be a non-empty string")
        if any(existing == name for existing, _, _ in self._blobs):
            raise ValueError(f"Duplicate blob name {name!r}; the reader keys blobs by name")
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError(f"Blob data must be bytes or bytearray, got {type(data)}")
        self._blobs.append((name, bytes(data), content_type))
        return self

    def _plan_layout(self) -> Tuple[bytes, int, List[Tuple[IgniteBlobEntry, bytes, int]]]:
        """Fixed-point layout of manifest and blobs.

        The blob directory lives inside the manifest, and every blob offset
        depends on the manifest's padded size, which depends on the digits of
        those offsets. The padded size is only ever grown, so the loop
        terminates; a single re-layout could leave the manifest longer than
        its pad and overrun the first blob.
        """
        base_manifest = {k: v for k, v in self.manifest.items() if k != "blobs"}
        crcs = [zlib.crc32(data) & 0xFFFFFFFF for _, data, _ in self._blobs]
        manifest_padded = align_up(len(json.dumps(base_manifest, indent=2).encode("utf-8")), ALIGNMENT)
        for _ in range(32):
            cur_offset = HEADER_SIZE + manifest_padded
            payloads: List[Tuple[IgniteBlobEntry, bytes, int]] = []
            for (name, data, ctype), crc in zip(self._blobs, crcs):
                cur_offset = align_up(cur_offset, ALIGNMENT)
                entry = IgniteBlobEntry(name=name, offset=cur_offset, size=len(data),
                                        content_type=ctype, crc32=crc)
                pad_len = align_up(len(data), ALIGNMENT) - len(data)
                payloads.append((entry, data, pad_len))
                cur_offset += len(data) + pad_len
            manifest = dict(base_manifest)
            manifest["blobs"] = [entry.to_dict() for entry, _, _ in payloads]
            manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")
            needed = align_up(len(manifest_bytes), ALIGNMENT)
            if needed <= manifest_padded:
                self.manifest = manifest
                return manifest_bytes, manifest_padded, payloads
            manifest_padded = needed
        raise RuntimeError("container layout did not converge")

    def write(self, output_path: Union[str, Path]) -> int:
        """
        Writes out the complete .ignite container file:
          1. Computes layout with strict 64-byte alignment.
          2. Encodes manifest JSON.
          3. Writes header, manifest, and aligned blobs.
          4. Computes CRC32 checksum over the body (offset 64 to EOF) and updates header.
        Returns total bytes written.
        """
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        # 1. Layout: manifest (with embedded blob directory) and 64-byte aligned blobs
        final_manifest_bytes, manifest_padded_size, blob_payloads = self._plan_layout()
        manifest_pad_len = manifest_padded_size - len(final_manifest_bytes)
        assert manifest_pad_len >= 0, "manifest overruns its padded section"
        blob_section_offset = HEADER_SIZE + manifest_padded_size
        total_file_size = blob_section_offset + sum(len(d) + pad for _, d, pad in blob_payloads)
        total_blob_size = total_file_size - blob_section_offset

        # 2. Write file
        with open(out_p, "wb") as f:
            # Provisional header
            hdr = IgniteHeader(
                arch_id=self.arch_id,
                total_file_size=total_file_size,
                manifest_offset=HEADER_SIZE,
                manifest_size=len(final_manifest_bytes),
                blob_offset=blob_section_offset,
                blob_size=total_blob_size,
                num_blobs=len(blob_payloads),
            )
            f.write(hdr.pack())

            # Manifest + padding
            f.write(final_manifest_bytes)
            if manifest_pad_len > 0:
                f.write(b"\x00" * manifest_pad_len)

            # Blobs + padding
            for entry, data, pad_len in blob_payloads:
                f.write(data)
                if pad_len > 0:
                    f.write(b"\x00" * pad_len)

        # 3. Calculate CRC32 of everything from offset 64 to end of file
        with open(out_p, "rb") as f:
            f.seek(HEADER_SIZE)
            body = f.read()
            body_crc = zlib.crc32(body) & 0xFFFFFFFF

        # 4. Update header with final CRC32
        hdr.crc32 = body_crc
        with open(out_p, "r+b") as f:
            f.seek(0)
            f.write(hdr.pack())

        return total_file_size


class IgniteModelReader:
    """
    Zero-copy memory-mapped reader for .ignite binary containers.
    Provides direct memoryview slices into 64-byte aligned transaction
    blobs and weight arrays.
    """

    def __init__(self, file_path_or_bytes: Union[str, Path, bytes, mmap.mmap]):
        self._file_handle: Optional[Any] = None
        self._mmap: Optional[mmap.mmap] = None
        self._buffer: Optional[Union[bytes, memoryview]] = None

        if isinstance(file_path_or_bytes, (str, Path)):
            self.path = Path(file_path_or_bytes)
            if not self.path.exists():
                raise FileNotFoundError(f".ignite model container not found: {self.path}")
            self._file_handle = open(self.path, "rb")
            self._mmap = mmap.mmap(self._file_handle.fileno(), 0, access=mmap.ACCESS_READ)
            self._buffer = memoryview(self._mmap)
            self.total_size = len(self._mmap)
        elif isinstance(file_path_or_bytes, mmap.mmap):
            self.path = None
            self._mmap = file_path_or_bytes
            self._buffer = memoryview(self._mmap)
            self.total_size = len(self._mmap)
        elif isinstance(file_path_or_bytes, (bytes, bytearray, memoryview)):
            self.path = None
            self._buffer = memoryview(file_path_or_bytes)
            self.total_size = len(self._buffer)
        else:
            raise TypeError(f"Unsupported input type for IgniteModelReader: {type(file_path_or_bytes)}")

        try:
            # Parse header. Every offset/size below is checked against the
            # buffer before it is used, so a truncated or corrupted directory
            # fails here instead of producing a short memoryview later.
            self.header = IgniteHeader.unpack(bytes(self._buffer[:HEADER_SIZE]))
            if self.header.total_file_size != self.total_size:
                raise ValueError(
                    f"Container declares {self.header.total_file_size} bytes but the buffer holds "
                    f"{self.total_size} (truncated or trailing data)")

            # Parse manifest
            m_start = self.header.manifest_offset
            m_end = m_start + self.header.manifest_size
            if m_end > self.total_size:
                raise ValueError(f"Manifest range [{m_start}, {m_end}) exceeds the container")
            manifest_raw = bytes(self._buffer[m_start:m_end]).decode("utf-8")
            self.manifest: Dict[str, Any] = json.loads(manifest_raw)

            # Build blob directory
            self.blobs: Dict[str, IgniteBlobEntry] = {}
            for b_dict in self.manifest.get("blobs", []):
                entry = IgniteBlobEntry.from_dict(b_dict)
                if entry.name in self.blobs:
                    raise ValueError(f"Duplicate blob name {entry.name!r} in the container directory")
                if entry.offset % ALIGNMENT != 0:
                    raise ValueError(f"Blob {entry.name!r} at offset {entry.offset} is not {ALIGNMENT}-byte aligned")
                if entry.offset < m_end or entry.size < 0 or entry.offset + entry.size > self.total_size:
                    raise ValueError(
                        f"Blob {entry.name!r} range [{entry.offset}, {entry.offset + entry.size}) "
                        f"is outside the blob section")
                self.blobs[entry.name] = entry
        except Exception:
            self.close()
            raise

    def verify_checksum(self) -> bool:
        """Verifies that the container's CRC32 checksum matches the body.

        The CRC covers offset 64 to the end of the file; the header's own
        fields are validated structurally at load, not by this checksum.
        """
        body = self._buffer[HEADER_SIZE : self.header.total_file_size]
        computed_crc = zlib.crc32(body) & 0xFFFFFFFF
        return computed_crc == self.header.crc32

    def verify_blob(self, name: str) -> bool:
        """Verifies one blob against the per-blob CRC32 stored in the directory."""
        entry = self.blobs[name]
        return (zlib.crc32(self.get_blob_memoryview(name)) & 0xFFFFFFFF) == entry.crc32

    def verify_all_blobs(self) -> Dict[str, bool]:
        return {name: self.verify_blob(name) for name in self.blobs}

    def get_blob_memoryview(self, name: str) -> memoryview:
        """
        Returns a zero-copy memoryview slice into the 64-byte aligned blob.
        Ideal for zero-memcpy binding to PyXRT instruction and host buffers.
        """
        if name not in self.blobs:
            raise KeyError(f"Blob '{name}' not found in container directory. Available: {list(self.blobs.keys())}")
        entry = self.blobs[name]
        start = entry.offset
        end = start + entry.size
        # The directory was validated at load; keep the guards so a mutated
        # entry cannot yield a short or misaligned view.
        if start % ALIGNMENT != 0:
            raise RuntimeError(f"Blob '{name}' is at offset {start}, which is not {ALIGNMENT}-byte aligned!")
        if end > self.total_size:
            raise RuntimeError(f"Blob '{name}' range [{start}, {end}) exceeds the container ({self.total_size} bytes)")
        return self._buffer[start:end]

    def get_blob_bytes(self, name: str) -> bytes:
        """Returns blob content as bytes."""
        return bytes(self.get_blob_memoryview(name))

    @property
    def is_dfl_fused(self) -> bool:
        """Indicates whether on-die AIE2 DFL decode micro-kernel is fused in this container."""
        return bool(self.manifest.get("fused_dfl", False))

    def close(self):
        """Closes memory map and underlying file handle."""
        if self._buffer is not None:
            try:
                self._buffer.release()
            except Exception:
                pass
            self._buffer = None
        if self._mmap is not None:
            try:
                self._mmap.close()
            except Exception:
                pass
            self._mmap = None
        if self._file_handle is not None:
            try:
                self._file_handle.close()
            except Exception:
                pass
            self._file_handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def decouple_container_weights(
    ignite_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    weights_path: Optional[Union[str, Path]] = None,
) -> Tuple[Path, Path]:
    """Decouples static weights from an existing monolithic .ignite container.

    Extracts 'wpackets.bin' into a separate sidecar file (.weights), records the
    weights SHA-256 and byte length in the container manifest, and writes out a
    slimmed container without the weights payload.

    Args:
        ignite_path: Path to existing monolithic .ignite file.
        output_path: Path for slimmed container. If None, appends '_decoupled.ignite'.
                     If identical to ignite_path, the container is safely overwritten.
        weights_path: Path for .weights sidecar file. If None, uses output_path.with_suffix('.weights').

    Returns:
        Tuple of (slimmed_container_path, weights_sidecar_path).
    """
    src_p = Path(ignite_path).resolve()
    if not src_p.exists():
        raise FileNotFoundError(f"Source container not found: {src_p}")

    if output_path is None:
        out_p = src_p.with_name(f"{src_p.stem}_decoupled.ignite")
    else:
        out_p = Path(output_path).resolve()

    if weights_path is None:
        w_p = out_p.with_suffix(".weights")
    else:
        w_p = Path(weights_path).resolve()

    with IgniteModelReader(src_p) as reader:
        if "wpackets.bin" not in reader.blobs:
            raise ValueError(f"Container {src_p} does not contain 'wpackets.bin' (already decoupled?)")

        weights_bytes = reader.get_blob_bytes("wpackets.bin")
        weights_sha = hashlib.sha256(weights_bytes).hexdigest()

        # Update manifest
        manifest = dict(reader.manifest)
        if "graph_engine" not in manifest:
            manifest["graph_engine"] = {}
        manifest["graph_engine"]["decoupled_weights"] = True
        manifest["graph_engine"]["weights_file"] = w_p.name
        manifest["graph_engine"]["weights_sha256"] = weights_sha
        manifest["graph_engine"]["weights_bytes"] = len(weights_bytes)

        # Collect blobs excluding wpackets.bin
        blobs: List[Tuple[str, bytes, str]] = []
        for b_dict in manifest.get("blobs", []):
            name = b_dict["name"]
            if name == "wpackets.bin":
                continue
            ctype = b_dict.get("content_type", "raw")
            blobs.append((name, reader.get_blob_bytes(name), ctype))
        arch_id = reader.header.arch_id

    # reader is closed here - avoiding any file locking on Windows
    w_p.parent.mkdir(parents=True, exist_ok=True)
    w_p.write_bytes(weights_bytes)

    writer = IgniteModelWriter(manifest_meta=manifest, arch_id=arch_id)
    for name, data, ctype in blobs:
        writer.add_blob(name, data, content_type=ctype)

    if out_p == src_p:
        tmp_p = out_p.with_suffix(".tmp.ignite")
        writer.write(tmp_p)
        tmp_p.replace(out_p)
    else:
        writer.write(out_p)

    with IgniteModelReader(out_p) as test_reader:
        if not test_reader.verify_checksum():
            raise RuntimeError(f"Slimmed container {out_p} failed checksum verification")

    return out_p, w_p

