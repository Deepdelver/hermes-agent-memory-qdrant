import json
import os
import threading
import urllib.error
import urllib.request
from typing import Any, Dict, List

DEFAULT_DENSE_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_SPARSE_MODEL = "Qdrant/bm25"

#: How long to wait on the remote embedder, and how many times to retry. Sized for an in-cluster
#: llama.cpp on a shared GPU, where a cold model or a busy card can take a few seconds.
REMOTE_TIMEOUT_S = float(os.getenv("QDRANT_REMOTE_TIMEOUT_S", "20"))
REMOTE_RETRIES = int(os.getenv("QDRANT_REMOTE_RETRIES", "2"))


class RemoteDenseEmbedder:
    """Dense vectors from an OpenAI-compatible /v1/embeddings endpoint. Sparse stays local.

    Added so dense embeddings can come from the in-cluster GPU (gpustack-models `embed`,
    embeddinggemma-300m on the shared card) instead of a local ONNX model, while the BM25 sparse
    half keeps running through fastembed.

    The sparse half is kept deliberately. This plugin's retrieval is hybrid -- dense vector plus
    BM25 sparse, fused by Qdrant -- and the GPU endpoint serves dense vectors only. Moving the
    whole embedder to it would silently drop the sparse arm and make recall worse on exactly the
    keyword-style queries BM25 is good at. One half of the embedder changing is a far smaller
    change to retrieval behaviour than both.

    NO LOCAL FALLBACK, deliberately. If the remote is unreachable this raises rather than quietly
    embedding with bge-small-en-v1.5: the two models differ in dimension (768 vs 384) and in
    geometry, so a fallback would write vectors the collection cannot accept -- or worse, write
    384-dim vectors into a 768-dim collection's sibling and surface later as an opaque Qdrant
    error. Failing the write is the recoverable outcome; a silent dimension change is not.
    """

    def __init__(self, base_url: str, model: str, timeout: float = REMOTE_TIMEOUT_S,
                 retries: int = REMOTE_RETRIES) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.retries = max(0, retries)
        self._dim: int | None = None
        self._sparse = None
        self._lock = threading.Lock()

    def _get_sparse(self):
        if self._sparse is None:
            with self._lock:
                if self._sparse is None:
                    from fastembed import SparseTextEmbedding

                    self._sparse = SparseTextEmbedding(DEFAULT_SPARSE_MODEL)
        return self._sparse

    def _post(self, texts: List[str]) -> List[List[float]]:
        body = json.dumps({"model": self.model, "input": texts}).encode("utf-8")
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            req = urllib.request.Request(
                f"{self.base_url}/embeddings", data=body,
                headers={"Content-Type": "application/json"}, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    payload = json.loads(resp.read().decode("utf-8", "replace"))
                rows = sorted(payload.get("data") or [], key=lambda r: r.get("index", 0))
                vectors = [r["embedding"] for r in rows]
                if len(vectors) != len(texts):
                    raise ValueError(
                        f"expected {len(texts)} embeddings, got {len(vectors)}")
                return vectors
            except Exception as exc:  # noqa: BLE001 - retried below, then raised
                last = exc
        raise RuntimeError(
            f"remote embedder {self.base_url} failed after {self.retries + 1} attempt(s): "
            f"{type(last).__name__}: {last}")

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = len(self.embed_one("dim probe"))
        return self._dim

    def warm(self) -> int:
        return self.dim

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._post([t if t else " " for t in texts])

    def embed_sparse(self, text: str) -> tuple[list[int], list[float]]:
        return self.embed_sparse_batch([text])[0]

    def embed_sparse_batch(self, texts: list[str]) -> list[tuple[list[int], list[float]]]:
        if not texts:
            return []
        clean = [t if t else " " for t in texts]
        out = []
        for result in self._get_sparse().embed(clean):
            if not result.indices.size:
                out.append(([0], [0.0]))
            else:
                out.append((result.indices.tolist(), result.values.tolist()))
        return out


class FastEmbedEmbedder:
    def __init__(
        self,
        model_name: str = DEFAULT_DENSE_MODEL,
        *,
        sparse_model_name: str = DEFAULT_SPARSE_MODEL,
        **_: Any,
    ) -> None:
        self.model_name = model_name or DEFAULT_DENSE_MODEL
        self.sparse_model_name = sparse_model_name or DEFAULT_SPARSE_MODEL
        self._dense = None
        self._sparse = None
        self._dim: int | None = None
        self._lock = threading.Lock()

    def _get_dense(self):
        if self._dense is None:
            with self._lock:
                if self._dense is None:
                    from fastembed import TextEmbedding

                    self._dense = TextEmbedding(self.model_name)
        return self._dense

    def _get_sparse(self):
        if self._sparse is None:
            with self._lock:
                if self._sparse is None:
                    from fastembed import SparseTextEmbedding

                    self._sparse = SparseTextEmbedding(self.sparse_model_name)
        return self._sparse

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = len(self.embed_one("dim probe"))
        return self._dim

    def warm(self) -> int:
        return self.dim

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        clean = [t if t else " " for t in texts]
        return [v.tolist() for v in self._get_dense().embed(clean)]

    def embed_sparse(self, text: str) -> tuple[list[int], list[float]]:
        return self.embed_sparse_batch([text])[0]

    def embed_sparse_batch(self, texts: list[str]) -> list[tuple[list[int], list[float]]]:
        if not texts:
            return []
        clean = [t if t else " " for t in texts]
        out = []
        for result in self._get_sparse().embed(clean):
            if not result.indices.size:
                out.append(([0], [0.0]))
            else:
                out.append((result.indices.tolist(), result.values.tolist()))
        return out


def embedder_from_config(embedding_cfg: Dict[str, Any] | None):
    """Build the embedder described by config.

    `base_url` selects the GPU embedder; its absence keeps the original local-ONNX behaviour, so
    an existing deployment that sets nothing is unaffected by this change.
    """
    cfg = embedding_cfg or {}
    base_url = (cfg.get("base_url") or "").strip()
    if base_url:
        return RemoteDenseEmbedder(base_url, cfg.get("remote_model") or "embeddinggemma-300m")
    return FastEmbedEmbedder(
        cfg.get("model", DEFAULT_DENSE_MODEL),
        sparse_model_name=cfg.get("sparse_model", DEFAULT_SPARSE_MODEL),
    )
