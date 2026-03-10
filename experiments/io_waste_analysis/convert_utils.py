"""
convert_utils.py — Utilitaires partagés pour le chargement de fichiers binaires DiskANN
========================================================================================
"""

import struct
import numpy as np


def load_diskann_bin(path: str) -> np.ndarray:
    """Charge un fichier .bin au format DiskANN et retourne un np.ndarray (npoints, ndims).

    Format attendu (conforme à diskann-utils/src/io.rs) :
      Offset 0-3  : npoints (u32, little-endian)
      Offset 4-7  : ndims   (u32, little-endian)
      Offset 8+   : npoints × ndims float32 (little-endian, row-major)
    """
    with open(path, "rb") as f:
        npoints, ndims = struct.unpack("<II", f.read(8))
        data = np.frombuffer(f.read(npoints * ndims * 4), dtype=np.float32)
    return data.reshape(npoints, ndims)
