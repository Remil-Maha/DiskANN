#!/usr/bin/env python3
"""
01_convert_npy_to_bin.py — Conversion .npy → format binaire DiskANN
====================================================================

POURQUOI CETTE CONVERSION ?
----------------------------
DiskANN (Rust) attend un format binaire spécifique :
  [npoints: u32 LE][ndims: u32 LE][data: npoints × ndims × sizeof(T)]

Ce format est défini dans diskann-utils/src/io.rs :
  - 8 octets d'en-tête : npoints (u32 little-endian) + ndims (u32 little-endian)
  - Payload : npoints × ndims éléments de T, tightly packed, row-major

NumPy utilise un format différent (.npy avec son propre header). On doit
convertir pour que les outils DiskANN puissent lire les données.

POURQUOI float32 ?
-------------------
DiskANN utilise f32 (float32) comme type de vecteur par défaut.
Les embeddings de modèles comme DPR, ANCE, etc. sont déjà en float32.
Si vos embeddings sont en float64, on les convertit en float32 (perte
négligeable pour ANN search).

USAGE :
-------
  python 01_convert_npy_to_bin.py \\
    --input  /chemin/vers/corpus_embeddings.npy \\
    --output data/nq_vectors.bin

  # Pour les requêtes aussi :
  python 01_convert_npy_to_bin.py \\
    --input  /chemin/vers/query_embeddings.npy \\
    --output data/nq_queries.bin
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np


def convert_npy_to_diskann_bin(input_path: str, output_path: str) -> None:
    """Convertit un fichier .npy en format binaire DiskANN.
    
    Format de sortie (identique à diskann-utils/src/io.rs) :
      Offset 0-3  : npoints (u32, little-endian)
      Offset 4-7  : ndims   (u32, little-endian)
      Offset 8+   : npoints × ndims float32 (little-endian, row-major)
    """
    print(f"Chargement de {input_path} ...")
    data = np.load(input_path)

    if data.ndim != 2:
        print(f"ERREUR : attendu un tableau 2D (npoints, ndims), obtenu shape={data.shape}")
        sys.exit(1)

    npoints, ndims = data.shape
    print(f"  Shape: {npoints:,} vecteurs × {ndims} dimensions")

    # Conversion en float32 si nécessaire
    # POURQUOI : DiskANN interne travaille en f32. Convertir maintenant évite
    # des surprises lors du chargement par le code Rust.
    if data.dtype != np.float32:
        print(f"  Conversion {data.dtype} → float32")
        data = data.astype(np.float32)

    # Vérification que les valeurs ne sont pas corrompues
    if np.any(np.isnan(data)):
        print("  ATTENTION : des NaN détectés dans les données !")
    if np.any(np.isinf(data)):
        print("  ATTENTION : des Inf détectés dans les données !")

    # Créer le dossier de sortie si nécessaire
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Écriture au format DiskANN
    # '<' = little-endian, 'II' = deux unsigned int 32-bit
    with open(output_path, "wb") as f:
        header = struct.pack("<II", npoints, ndims)
        f.write(header)
        # Le .tobytes() de numpy produit du row-major (C order) par défaut,
        # ce qui correspond au layout attendu par DiskANN.
        # On force C-contiguous pour être sûr.
        f.write(np.ascontiguousarray(data).tobytes())

    file_size = Path(output_path).stat().st_size
    expected_size = 8 + npoints * ndims * 4  # 8 bytes header + float32 data
    assert file_size == expected_size, (
        f"Taille fichier {file_size} != attendue {expected_size}"
    )

    print(f"  Écrit : {output_path} ({file_size:,} octets)")
    print(f"  Header : npoints={npoints}, ndims={ndims}")
    print(f"  Vérification : OK ✓")


def load_diskann_bin(path: str) -> np.ndarray:
    """Charge un fichier .bin au format DiskANN et retourne un np.ndarray (npoints, ndims).
    
    POURQUOI CETTE FONCTION ICI ?
    Utilitaire pratique pour vérifier que la conversion s'est bien passée
    et réutilisé par les scripts suivants.
    """
    with open(path, "rb") as f:
        npoints, ndims = struct.unpack("<II", f.read(8))
        data = np.frombuffer(f.read(npoints * ndims * 4), dtype=np.float32)
    return data.reshape(npoints, ndims)


def main():
    parser = argparse.ArgumentParser(
        description="Convertit un fichier .npy en format binaire DiskANN"
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Chemin vers le fichier .npy (embeddings, shape [N, D])"
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Chemin du fichier .bin de sortie (format DiskANN)"
    )
    parser.add_argument(
        "--verify", action="store_true", default=True,
        help="Vérifier la conversion en relisant le fichier (défaut: oui)"
    )
    args = parser.parse_args()

    convert_npy_to_diskann_bin(args.input, args.output)

    if args.verify:
        print("\nVérification par relecture :")
        original = np.load(args.input).astype(np.float32)
        reloaded = load_diskann_bin(args.output)
        assert original.shape == reloaded.shape, (
            f"Shape mismatch: {original.shape} vs {reloaded.shape}"
        )
        assert np.allclose(original, reloaded), "Données différentes !"
        print(f"  Relecture OK : {reloaded.shape[0]:,} × {reloaded.shape[1]} ✓")


if __name__ == "__main__":
    main()
