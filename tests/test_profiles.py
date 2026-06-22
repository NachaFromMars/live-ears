from scripts.profiles import ProfileStore, VoiceProfile
import numpy as np
import tempfile
from pathlib import Path

temp_dir = Path(tempfile.mkdtemp())
store = ProfileStore(profiles_dir=temp_dir)
print(f'Store: {store}')

emb1 = np.random.randn(192).astype(np.float32)
profile = store.save('Nang', emb1)
print(f'Saved: {profile.name} ({profile.id})')

loaded = store.load('Nang')
print(f'Loaded shape: {loaded.shape}')
print(f'Match: {np.allclose(emb1, loaded)}')

print(f'Exists Nang: {store.exists("Nang")}')
print(f'Exists Minh: {store.exists("Minh")}')

profiles = store.list()
print(f'Profiles: {[p.name for p in profiles]}')

emb2 = np.random.randn(192).astype(np.float32)
store.save('Minh', emb2)
print(f'Count: {store.count()}')

store.rename('Minh', 'Tuan')
print(f'After rename: {[p.name for p in store.list()]}')

store.delete('Tuan')
print(f'After delete: {store.count()}')

import shutil
shutil.rmtree(temp_dir)

print('Step 4.2 PASS')
