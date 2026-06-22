"""
Identifier — Speaker identification via voice embeddings.

Matches audio against enrolled profiles, auto-saves unknowns.
"""

import sys
from typing import Optional, Tuple, List
from datetime import datetime

import numpy as np

from .config import ID_THRESHOLD, ENROLL_DURATION_S, PROFILES_DIR
from .embedder import VoiceEmbedder
from .profiles import ProfileStore, VoiceProfile


class SpeakerIdentifier:
    """
    Speaker identification system.
    
    Usage:
        identifier = SpeakerIdentifier()
        identifier.load()
        
        # Identify
        name, confidence = identifier.identify(audio_array)
        
        # Enroll
        identifier.enroll("Nấng", audio_array)
    """

    def __init__(
        self,
        threshold: float = ID_THRESHOLD,
        auto_save_unknown: bool = True,
    ):
        self.threshold = threshold
        self.auto_save_unknown = auto_save_unknown
        
        self._embedder = VoiceEmbedder()
        self._store = ProfileStore(PROFILES_DIR)
        self._loaded = False
        self._unknown_counter = 0

    def load(self) -> None:
        """Load embedder model."""
        if self._loaded:
            return
        self._embedder.load()
        self._loaded = True

    def ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    # ── Identification ───────────────────────────────────────────

    def identify(self, audio: np.ndarray) -> Tuple[str, float]:
        """
        Identify speaker from audio.
        
        Returns:
            (name, confidence) — confidence is cosine similarity
        """
        self.ensure_loaded()

        # Extract embedding
        query_emb = self._embedder.extract(audio)

        # Load all profiles
        profiles = self._store.load_all()

        if not profiles:
            # No enrolled profiles
            if self.auto_save_unknown:
                name = self._save_unknown(query_emb)
                return name, 0.0
            return "Unknown", 0.0

        # Find best match
        best_name = "Unknown"
        best_score = -1.0

        for name, profile_emb in profiles.items():
            score = self._embedder.cosine_similarity(query_emb, profile_emb)
            if score > best_score:
                best_score = score
                best_name = name

        # Check threshold
        if best_score < self.threshold:
            if self.auto_save_unknown:
                name = self._save_unknown(query_emb)
                return name, float(best_score)
            return "Unknown", float(best_score)

        return best_name, float(best_score)

    def identify_batch(
        self,
        audios: List[np.ndarray],
    ) -> List[Tuple[str, float]]:
        """Identify multiple audio samples."""
        self.ensure_loaded()

        embeddings = self._embedder.extract_batch(audios)
        profiles = self._store.load_all()

        results = []
        for emb in embeddings:
            if not profiles:
                name = self._save_unknown(emb) if self.auto_save_unknown else "Unknown"
                results.append((name, 0.0))
                continue

            best_name = "Unknown"
            best_score = -1.0

            for pname, pemb in profiles.items():
                score = self._embedder.cosine_similarity(emb, pemb)
                if score > best_score:
                    best_score = score
                    best_name = pname

            if best_score < self.threshold:
                name = self._save_unknown(emb) if self.auto_save_unknown else "Unknown"
                results.append((name, float(best_score)))
            else:
                results.append((best_name, float(best_score)))

        return results

    # ── Enrollment ───────────────────────────────────────────────

    def enroll(
        self,
        name: str,
        audio: np.ndarray,
        overwrite: bool = False,
    ) -> VoiceProfile:
        """
        Enroll a new speaker.
        
        Args:
            name: Speaker name
            audio: Voice sample (≥5s recommended)
            overwrite: Replace existing profile
            
        Returns:
            VoiceProfile object
        """
        self.ensure_loaded()

        embedding = self._embedder.extract(audio)
        profile = self._store.save(name, embedding, overwrite=overwrite)

        print(f"[Identifier] Enrolled: {name}", file=sys.stderr)
        return profile

    def enroll_multiple(
        self,
        name: str,
        audios: List[np.ndarray],
        overwrite: bool = False,
    ) -> VoiceProfile:
        """
        Enroll from multiple audio samples (robust enrollment).
        
        Averages embeddings for better accuracy.
        """
        self.ensure_loaded()

        embeddings = self._embedder.extract_batch(audios)
        avg_embedding = self._embedder.average_embeddings(list(embeddings))

        profile = self._store.save(name, avg_embedding, overwrite=overwrite)
        print(f"[Identifier] Enrolled {name} from {len(audios)} samples", file=sys.stderr)
        return profile

    # ── Unknown Handling ─────────────────────────────────────────

    def _save_unknown(self, embedding: np.ndarray) -> str:
        """Auto-save unknown speaker embedding."""
        self._unknown_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"Unknown_{timestamp}_{self._unknown_counter}"

        try:
            self._store.save(name, embedding)
            print(f"[Identifier] Auto-saved unknown: {name}", file=sys.stderr)
        except Exception as e:
            print(f"[Identifier] Failed to save unknown: {e}", file=sys.stderr)

        return name

    # ── Profile Management ───────────────────────────────────────

    def list_profiles(self) -> List[VoiceProfile]:
        """List all enrolled profiles."""
        return self._store.list()

    def delete_profile(self, name: str) -> None:
        """Delete a profile."""
        self._store.delete(name)

    def rename_profile(self, old_name: str, new_name: str) -> None:
        """Rename a profile."""
        self._store.rename(old_name, new_name)

    def profile_exists(self, name: str) -> bool:
        """Check if profile exists."""
        return self._store.exists(name)

    def count_profiles(self) -> int:
        """Count enrolled profiles."""
        return self._store.count()

    # ── Verification ─────────────────────────────────────────────

    def verify(
        self,
        audio: np.ndarray,
        claimed_name: str,
    ) -> Tuple[bool, float]:
        """
        Verify if audio belongs to claimed speaker.
        
        Returns:
            (is_match, confidence)
        """
        self.ensure_loaded()

        if not self._store.exists(claimed_name):
            return False, 0.0

        query_emb = self._embedder.extract(audio)
        profile_emb = self._store.load(claimed_name)

        score = self._embedder.cosine_similarity(query_emb, profile_emb)
        is_match = score >= self.threshold

        return is_match, float(score)

    # ── Properties ───────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not loaded"
        return f"SpeakerIdentifier({self.count_profiles()} profiles, {status})"
