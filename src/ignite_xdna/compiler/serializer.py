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
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError(f"Blob data must be bytes or bytearray, got {type(data)}")
        self._blobs.append((name, bytes(data), content_type))
        return self

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

        # 1. First pass: layout planning with provisional manifest
        manifest_json_bytes = json.dumps(self.manifest, indent=2).encode("utf-8")
        manifest_size = len(manifest_json_bytes)
        manifest_padded_size = align_up(manifest_size, ALIGNMENT)

        blob_section_offset = HEADER_SIZE + manifest_padded_size

        # Plan blob offsets
        blob_entries: List[IgniteBlobEntry] = []
        cur_offset = blob_section_offset

        blob_payloads: List[Tuple[IgniteBlobEntry, bytes, int]] = []
        for name, data, ctype in self._blobs:
            cur_offset = align_up(cur_offset, ALIGNMENT)
            b_size = len(data)
            b_crc = zlib.crc32(data) & 0xFFFFFFFF
            entry = IgniteBlobEntry(
                name=name,
                offset=cur_offset,
                size=b_size,
                content_type=ctype,
                crc32=b_crc,
            )
            blob_entries.append(entry)
            pad_len = align_up(b_size, ALIGNMENT) - b_size
            blob_payloads.append((entry, data, pad_len))
            cur_offset += b_size + pad_len

        # Embed blob table into manifest
        self.manifest["blobs"] = [e.to_dict() for e in blob_entries]
        final_manifest_bytes = json.dumps(self.manifest, indent=2).encode("utf-8")

        # If embedding blob directory changed manifest padded size, recalculate offsets
        if len(final_manifest_bytes) > manifest_padded_size:
            manifest_padded_size = align_up(len(final_manifest_bytes), ALIGNMENT)
            blob_section_offset = HEADER_SIZE + manifest_padded_size
            cur_offset = blob_section_offset
            blob_entries.clear()
            blob_payloads.clear()
            for name, data, ctype in self._blobs:
                cur_offset = align_up(cur_offset, ALIGNMENT)
                b_size = len(data)
                b_crc = zlib.crc32(data) & 0xFFFFFFFF
                entry = IgniteBlobEntry(
                    name=name,
                    offset=cur_offset,
                    size=b_size,
                    content_type=ctype,
                    crc32=b_crc,
                )
                blob_entries.append(entry)
                pad_len = align_up(b_size, ALIGNMENT) - b_size
                blob_payloads.append((entry, data, pad_len))
                cur_offset += b_size + pad_len
            self.manifest["blobs"] = [e.to_dict() for e in blob_entries]
            final_manifest_bytes = json.dumps(self.manifest, indent=2).encode("utf-8")

        manifest_pad_len = manifest_padded_size - len(final_manifest_bytes)
        total_blob_size = cur_offset - blob_section_offset
        total_file_size = cur_offset

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
                num_blobs=len(blob_entries),
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
            # Parse header
            self.header = IgniteHeader.unpack(bytes(self._buffer[:HEADER_SIZE]))

            # Parse manifest
            m_start = self.header.manifest_offset
            m_end = m_start + self.header.manifest_size
            manifest_raw = bytes(self._buffer[m_start:m_end]).decode("utf-8")
            self.manifest: Dict[str, Any] = json.loads(manifest_raw)

            # Build blob directory
            self.blobs: Dict[str, IgniteBlobEntry] = {}
            for b_dict in self.manifest.get("blobs", []):
                entry = IgniteBlobEntry.from_dict(b_dict)
                self.blobs[entry.name] = entry
        except Exception:
            self.close()
            raise

    def verify_checksum(self) -> bool:
        """Verifies that the container's CRC32 checksum matches the body."""
        body = self._buffer[HEADER_SIZE : self.header.total_file_size]
        computed_crc = zlib.crc32(body) & 0xFFFFFFFF
        return computed_crc == self.header.crc32

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
        # Assert strict 64-byte alignment
        if start % ALIGNMENT != 0:
            raise RuntimeError(f"Blob '{name}' is at offset {start}, which is not {ALIGNMENT}-byte aligned!")
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
