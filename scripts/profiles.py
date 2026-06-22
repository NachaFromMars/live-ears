"""
ProfileStore — Voice profile storage and management.

Stores voice embeddings as .npy files with JSON manifest for metadata.
Supports save, load, delete, rename, list operations.
"""

import sys
import json
import time
from pathlib import Path
from typing import Optional, List, Dict
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np

from .config import PROFILES_DIR, MANIFEST_FILE, ID_EMBEDDING_DIM


@dataclass
class VoiceProfile:
    """A voice profile entry."""
    id: str                    # Unique ID (filename without .npy)
    name: str                  # Display name
    created: str               # ISO timestamp
    embedding_file: str        # Filename of .npy file
    sample_count: int = 1      # Number of samples averaged
    updated: Optional[str] = None  # Last update timestamp

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "VoiceProfile":
        return cls(**data)


class ProfileStore:
    """
    Voice profile storage manager.
    
    Usage:
        store = ProfileStore()
        
        # Save embedding
        profile = store.save("Nấng", embedding_array)
        
        # Load embedding
        emb = store.load("Nấng")
        
        # List profiles
        profiles = store.list()
        
        # Delete
        store.delete("Nấng")
        
        # Rename
        store.rename("Unknown_01", "Tuấn")
    """

    def __init__(self, profiles_dir: Path = PROFILES_DIR):
        self.profiles_dir = Path(profiles_dir)
        self.manifest_file = MANIFEST_FILE
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self._manifest: Dict[str, VoiceProfile] = {}
        self._load_manifest()

    # ── Manifest Management ──────────────────────────────────────

    def _load_manifest(self) -> None:
        """Load manifest from disk."""
        if not self.manifest_file.exists():
            self._manifest = {}
            self._save_manifest()
            return

        try:
            with open(self.manifest_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            self._manifest = {
                k: VoiceProfile.from_dict(v) for k, v in data.items()
            }
        except Exception as e:
            print(f"[ProfileStore] Manifest load error: {e}", file=sys.stderr)
            self._manifest = {}

    def _save_manifest(self) -> None:
        """Save manifest to disk."""
        data = {k: v.to_dict() for k, v in self._manifest.items()}
        
        with open(self.manifest_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # ── Save / Load ──────────────────────────────────────────────

    def save(
        self,
        name: str,
        embedding: np.ndarray,
        overwrite: bool = False,
    ) -> VoiceProfile:
        """
        Save voice embedding with name.
        
        Args:
            name: Profile name (will be sanitized for filename)
            embedding: 192-dim numpy array
            overwrite: If True, overwrite existing profile
            
        Returns:
            VoiceProfile object
        """
        if embedding.shape != (ID_EMBEDDING_DIM,):
            raise ValueError(f"Embedding must be {ID_EMBEDDING_DIM}-dim, got {embedding.shape}")

        # Sanitize name for filename
        profile_id = self._sanitize_name(name)
        embedding_file = f"{profile_id}.npy"
        embedding_path = self.profiles_dir / embedding_file

        # Check if exists
        if profile_id in self._manifest and not overwrite:
            raise FileExistsError(f"Profile '{name}' already exists. Use overwrite=True to replace.")

        # Save embedding
        np.save(embedding_path, embedding)

        # Create/update profile
        now = datetime.now().isoformat()
        
        if profile_id in self._manifest:
            profile = self._manifest[profile_id]
            profile.updated = now
            profile.sample_count += 1
        else:
            profile = VoiceProfile(
                id=profile_id,
                name=name,
                created=now,
                embedding_file=embedding_file,
            )

        self._manifest[profile_id] = profile
        self._save_manifest()

        print(f"[ProfileStore] Saved profile: {name} ({profile_id})", file=sys.stderr)
        return profile

    def load(self, name: str) -> np.ndarray:
        """
        Load embedding by name.
        
        Args:
            name: Profile name or ID
            
        Returns:
            192-dim numpy array
        """
        profile_id = self._sanitize_name(name)

        if profile_id not in self._manifest:
            raise KeyError(f"Profile not found: {name}")

        profile = self._manifest[profile_id]
        embedding_path = self.profiles_dir / profile.embedding_file

        if not embedding_path.exists():
            raise FileNotFoundError(f"Embedding file missing: {embedding_path}")

        embedding = np.load(embedding_path)
        return embedding.astype(np.float32)

    def load_all(self) -> Dict[str, np.ndarray]:
        """Load all embeddings as {name: embedding} dict."""
        embeddings = {}
        for profile_id, profile in self._manifest.items():
            try:
                embeddings[profile.name] = self.load(profile_id)
            except Exception as e:
                print(f"[ProfileStore] Failed to load {profile.name}: {e}", file=sys.stderr)
        return embeddings

    # ── Delete / Rename ──────────────────────────────────────────

    def delete(self, name: str) -> None:
        """Delete a profile."""
        profile_id = self._sanitize_name(name)

        if profile_id not in self._manifest:
            raise KeyError(f"Profile not found: {name}")

        profile = self._manifest[profile_id]
        embedding_path = self.profiles_dir / profile.embedding_file

        # Delete file
        if embedding_path.exists():
            embedding_path.unlink()

        # Remove from manifest
        del self._manifest[profile_id]
        self._save_manifest()

        print(f"[ProfileStore] Deleted profile: {name}", file=sys.stderr)

    def rename(self, old_name: str, new_name: str) -> None:
        """Rename a profile."""
        old_id = self._sanitize_name(old_name)

        if old_id not in self._manifest:
            raise KeyError(f"Profile not found: {old_name}")

        new_id = self._sanitize_name(new_name)

        if new_id in self._manifest and new_id != old_id:
            raise FileExistsError(f"Profile '{new_name}' already exists")

        profile = self._manifest[old_id]

        # Rename file if ID changed
        if old_id != new_id:
            old_path = self.profiles_dir / profile.embedding_file
            new_file = f"{new_id}.npy"
            new_path = self.profiles_dir / new_file

            if old_path.exists():
                old_path.rename(new_path)

            profile.embedding_file = new_file
            del self._manifest[old_id]

        # Update name
        profile.name = new_name
        profile.id = new_id
        profile.updated = datetime.now().isoformat()

        self._manifest[new_id] = profile
        self._save_manifest()

        print(f"[ProfileStore] Renamed: {old_name} → {new_name}", file=sys.stderr)

    # ── List / Query ─────────────────────────────────────────────

    def list(self) -> List[VoiceProfile]:
        """List all profiles."""
        return list(self._manifest.values())

    def exists(self, name: str) -> bool:
        """Check if profile exists."""
        profile_id = self._sanitize_name(name)
        return profile_id in self._manifest

    def get_profile(self, name: str) -> Optional[VoiceProfile]:
        """Get profile metadata by name."""
        profile_id = self._sanitize_name(name)
        return self._manifest.get(profile_id)

    def count(self) -> int:
        """Count profiles."""
        return len(self._manifest)

    # ── Utilities ────────────────────────────────────────────────

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """Sanitize name for safe filename (lowercase, alphanumeric + underscore)."""
        import re
        sanitized = re.sub(r'[^a-zA-Z0-9_\-]', '_', name).lower()
        sanitized = re.sub(r'_+', '_', sanitized).strip('_')
        return sanitized or "unnamed"

    def clear(self) -> None:
        """Delete all profiles (use with caution)."""
        for profile in list(self._manifest.values()):
            self.delete(profile.id)

    def __repr__(self) -> str:
        return f"ProfileStore({self.count()} profiles)"
