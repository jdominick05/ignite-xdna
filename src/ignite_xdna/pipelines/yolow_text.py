"""YOLO-World v2's vocabulary at run time: CLIP text embeddings for class names, and the tensors they become.

A YOLO-World v2 container runs its convolutions on the NPU and its four text cross-attention regions on the host. The
vocabulary reaches the network in two places only:

- the attention regions' text guides, ``/model.{12,15,18,21}/attn/Reshape_output_0`` of shape (1, K, heads, 32): each
  region's guide projection applied to the L2-normalized CLIP embeddings of the K class names. The export folds them
  into initializers, which ``EngineSession.set_host_constants`` replaces;
- the contrastive decode, which scores each anchor's 512-channel visual feature against the same embeddings.

So another vocabulary needs no new container. The text side is a bundle ``pipelines/yolow/6_text_encoder.py`` writes
in the model-tools environment: CLIP ViT-B/32's text encoder as ONNX (int64 tokens [K, 77] -> normalized float32
[K, 512], run here on ONNX Runtime's CPU provider) and an npz with the tokenizer's BPE merges, the four guide
projections and the contrastive scales and biases.

The tokenizer here needs neither ``regex`` nor ``ftfy``. It matches CLIP's own on printable ASCII text and refuses
anything else rather than tokenize it differently.
"""
from __future__ import annotations

import html
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np

CONTEXT_LENGTH = 77
GUIDE_BLOCKS = (12, 15, 18, 21)
_PRINTABLE_ASCII = re.compile(r"[\x20-\x7e\t\n\r]*")
# CLIP's pre-split, <|startoftext|>|<|endoftext|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+,
# restricted to ASCII, where \p{L} is [a-z] after lower-casing and \p{N} is [0-9].
_SPLIT = re.compile(r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[a-z]+|[0-9]|[^\sa-z0-9]+")


def guide_name(block: int) -> str:
    return f"/model.{block}/attn/Reshape_output_0"


@lru_cache(maxsize=None)
def _bytes_to_unicode() -> Dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) + \
        list(range(ord("\xae"), ord("\xff") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


class ClipTokenizer:
    """CLIP's byte-level BPE tokenizer for printable ASCII text; ``merges`` are the BPE merge lines in rank order."""

    def __init__(self, merges: Sequence[str]):
        pairs = [tuple(m.split()) for m in merges]
        byte_symbols = list(_bytes_to_unicode().values())
        vocab = byte_symbols + [f"{s}</w>" for s in byte_symbols] + ["".join(p) for p in pairs]
        vocab += ["<|startoftext|>", "<|endoftext|>"]
        self.encoder = {tok: i for i, tok in enumerate(vocab)}
        self.ranks = {p: i for i, p in enumerate(pairs)}
        self.sot, self.eot = self.encoder["<|startoftext|>"], self.encoder["<|endoftext|>"]
        self._cache: Dict[str, List[str]] = {"<|startoftext|>": ["<|startoftext|>"], "<|endoftext|>": ["<|endoftext|>"]}

    def _bpe(self, token: str) -> List[str]:
        if token in self._cache:
            return self._cache[token]
        word = list(token[:-1]) + [token[-1] + "</w>"]
        while len(word) > 1:
            ranked = [(self.ranks[pair], i) for i, pair in enumerate(zip(word, word[1:])) if pair in self.ranks]
            if not ranked:
                break
            _, at = min(ranked)   # the pair with the lowest merge rank; every occurrence of it merges, left to right
            first, second = word[at], word[at + 1]
            merged, i = [], 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    merged.append(first + second)
                    i += 2
                else:
                    merged.append(word[i])
                    i += 1
            word = merged
        self._cache[token] = word
        return word

    def encode(self, text: str) -> List[int]:
        if not _PRINTABLE_ASCII.fullmatch(text):
            raise ValueError(f"only printable ASCII class names are supported, got {text!r}")
        text = html.unescape(html.unescape(text)).strip()
        if not _PRINTABLE_ASCII.fullmatch(text):
            raise ValueError(f"only printable ASCII class names are supported, got {text!r} after HTML unescaping")
        text = re.sub(r"\s+", " ", text).strip().lower()
        byte_map = _bytes_to_unicode()
        ids: List[int] = []
        for piece in _SPLIT.findall(text):
            ids.extend(self.encoder[t] for t in self._bpe("".join(byte_map[b] for b in piece.encode("utf-8"))))
        return ids

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        """int64 [len(texts), 77]: start token, the text, end token, zeros; a longer text is cut and ends in the end
        token (CLIP's ``truncate=True``)."""
        out = np.zeros((len(texts), CONTEXT_LENGTH), dtype=np.int64)
        for row, text in enumerate(texts):
            ids = [self.sot, *self.encode(text), self.eot]
            if len(ids) > CONTEXT_LENGTH:
                ids = ids[:CONTEXT_LENGTH]
                ids[-1] = self.eot
            out[row, :len(ids)] = ids
        return out


class YoloWorldText:
    """Class names -> CLIP embeddings, text guides and contrastive constants, from a ``6_text_encoder.py`` bundle."""

    def __init__(self, encoder_onnx: Union[str, Path], bundle_npz: Union[str, Path]):
        import onnxruntime as ort
        from ignite_xdna.pipelines.power import ort_session_options
        with np.load(bundle_npz, allow_pickle=False) as z:
            self.tokenizer = ClipTokenizer([str(m) for m in z["bpe_merges"]])
            self.projections = {b: (z[f"model.{b}.attn.gl.weight"].astype(np.float32),
                                    z[f"model.{b}.attn.gl.bias"].astype(np.float32), int(z[f"model.{b}.attn.hc"]))
                                for b in GUIDE_BLOCKS}
            self.contrastive_scales = tuple(float(v) for v in z["contrastive_scale"])
            self.contrastive_biases = tuple(float(v) for v in z["contrastive_bias"])
        self.session = ort.InferenceSession(str(encoder_onnx), ort_session_options(), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def embed(self, names: Sequence[str]) -> np.ndarray:
        """float32 [K, 512], L2-normalized CLIP ViT-B/32 text embeddings."""
        if not names:
            raise ValueError("the vocabulary is empty")
        return np.asarray(self.session.run(None, {self.input_name: self.tokenizer(names)})[0], dtype=np.float32)

    def host_constants(self, embeddings: np.ndarray) -> Dict[str, np.ndarray]:
        """The four attention regions' text guides for ``EngineSession.set_host_constants``."""
        e = np.asarray(embeddings, dtype=np.float32)
        out = {}
        for b, (w, bias, hc) in self.projections.items():
            g = e @ w.T + bias
            out[guide_name(b)] = np.ascontiguousarray(g.reshape(1, e.shape[0], g.shape[1] // hc, hc), dtype=np.float32)
        return out

    def vocabulary(self, names: Sequence[str]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        e = self.embed(names)
        return e, self.host_constants(e)
