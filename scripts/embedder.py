"""
Embedder — SpeechBrain ECAPA-TDNN voice embedding extractor.

Extracts 192-dimensional voice embeddings (fingerprints) from audio.
Used for speaker enrollment and identification.
"""

import sys
import time
from typing import Optional

import numpy as np
import torch

from .config import (
    SAMPLE_RATE,
    ID_EMBEDDING_DIM,
)


class VoiceEmbedder:
    """
    SpeechBrain ECAPA-TDNN voice embedding extractor.
    
    Extracts 192-dim speaker embeddings from audio for identification.
    
    Usage:
        embedder = VoiceEmbedder()
        embedder.load()
        
        embedding = embedder.extract(audio_array)
        # embedding.shape = (192,)
        
        similarity = embedder.cosine_similarity(emb1, emb2)
    """

    def __init__(self):
        self._model = None
        self._loaded = False

    # ── Model Loading ────────────────────────────────────────────

    def load(self) -> None:
        """Load SpeechBrain ECAPA-TDNN model from HuggingFace."""
        if self._loaded:
            return

        from speechbrain.inference.speaker import EncoderClassifier

        print("[VoiceEmbedder] Loading ECAPA-TDNN model...", file=sys.stderr)
        t0 = time.time()

        self._model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="models/ecapa-tdnn",  # Cache in skill dir
            run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        )

        elapsed = time.time() - t0
        print(f"[VoiceEmbedder] Loaded in {elapsed:.1f}s", file=sys.stderr)
        self._loaded = True

    def ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    # ── Embedding Extraction ─────────────────────────────────────

    def extract(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract voice embedding from audio.
        
        Args:
            audio: float32 mono 16kHz numpy array
            
        Returns:
            192-dim embedding as numpy array (float32)
        """
        self.ensure_loaded()

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # SpeechBrain expects torch tensor
        audio_tensor = torch.from_numpy(audio).unsqueeze(0)  # (1, samples)

        with torch.no_grad():
            embedding = self._model.encode_batch(audio_tensor)
            # embedding shape: (1, 1, 192)
            embedding = embedding.squeeze().cpu().numpy()

        return embedding.astype(np.float32)

    def extract_batch(self, audios: list[np.ndarray]) -> np.ndarray:
        """
        Extract embeddings for multiple audio samples.
        
        Args:
            audios: List of float32 mono 16kHz arrays
            
        Returns:
            (N, 192) array of embeddings
        """
        self.ensure_loaded()

        # Pad to same length
        max_len = max(len(a) for a in audios)
        padded = []
        for audio in audios:
            if len(audio) < max_len:
                pad = np.zeros(max_len, dtype=np.float32)
                pad[:len(audio)] = audio
                padded.append(pad)
            else:
                padded.append(audio)

        # Stack to batch
        batch = torch.from_numpy(np.stack(padded))  # (N, samples)

        with torch.no_grad():
            embeddings = self._model.encode_batch(batch)
            # shape: (N, 1, 192)
            embeddings = embeddings.squeeze(1).cpu().numpy()

        return embeddings.astype(np.float32)

    # ── Similarity ───────────────────────────────────────────────

    @staticmethod
    def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Returns:
            Similarity score in range [-1, 1], higher = more similar
        """
        dot = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot / (norm1 * norm2))

    @staticmethod
    def euclidean_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute Euclidean distance between two embeddings."""
        return float(np.linalg.norm(emb1 - emb2))

    def compare(self, emb1: np.ndarray, emb2: np.ndarray) -> dict:
        """
        Compare two embeddings with multiple metrics.
        
        Returns:
            Dict with cosine_similarity and euclidean_distance
        """
        return {
            "cosine_similarity": self.cosine_similarity(emb1, emb2),
            "euclidean_distance": self.euclidean_distance(emb1, emb2),
        }

    def verify(
        self,
        emb1: np.ndarray,
        emb2: np.ndarray,
        threshold: float = 0.75,
    ) -> bool:
        """
        Verify if two embeddings belong to the same speaker.
        
        Args:
            emb1, emb2: Voice embeddings
            threshold: Cosine similarity threshold (0.75 = balanced)
            
        Returns:
            True if same speaker, False otherwise
        """
        similarity = self.cosine_similarity(emb1, emb2)
        return similarity >= threshold

    def find_most_similar(
        self,
        query: np.ndarray,
        candidates: list[np.ndarray],
    ) -> tuple[int, float]:
        """
        Find most similar embedding in candidates.
        
        Args:
            query: Query embedding
            candidates: List of candidate embeddings
            
        Returns:
            (index, similarity) of best match
        """
        if not candidates:
            return -1, 0.0

        best_idx = -1
        best_sim = -1.0

        for i, cand in enumerate(candidates):
            sim = self.cosine_similarity(query, cand)
            if sim > best_sim:
                best_sim = sim
                best_idx = i

        return best_idx, float(best_sim)

    # ── Utilities ────────────────────────────────────────────────

    @staticmethod
    def normalize(embedding: np.ndarray) -> np.ndarray:
        """L2-normalize embedding (unit length)."""
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm

    @staticmethod
    def average_embeddings(embeddings: list[np.ndarray]) -> np.ndarray:
        """Average multiple embeddings (for robust enrollment)."""
        if not embeddings:
            return np.zeros(ID_EMBEDDING_DIM, dtype=np.float32)
        stacked = np.stack(embeddings)
        return np.mean(stacked, axis=0).astype(np.float32)

    # ── Properties ───────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def embedding_dim(self) -> int:
        return ID_EMBEDDING_DIM

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not loaded"
        return f"VoiceEmbedder(ECAPA-TDNN, {status})"
