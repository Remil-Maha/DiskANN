#!/usr/bin/env python3
"""
02_build_index.py — Construction d'un index disque DiskANN via le builder Rust
===============================================================================

POURQUOI LE BUILDER RUST ET PAS hnswlib ?
-------------------------------------------
L'objectif de l'expérience est de mesurer le gaspillage d'I/O quand on CHERCHE
dans un index DiskANN sur DISQUE. Pour cela il faut :

  1. Un VRAI index DiskANN avec le layout secteur-aligné (nœuds empaquetés
     dans des pages de 4096 octets)
  2. Le VRAI graphe de voisinage Vamana (avec le pruning α=1.2, saturation)
  3. Le VRAI format binaire que DiskANN lit pendant la recherche

hnswlib ne convient PAS car :
  - Il charge TOUT l'index en mémoire → pas de lecture disque → pas d'I/O à mesurer
  - Il construit un graphe HNSW (multi-couches) ≠ graphe Vamana (une couche, pruning α)
  - Il ne produit pas le layout secteur-aligné → pas de correspondance nœud↔secteur

Le builder Rust de DiskANN (DiskIndexBuilder) :
  - Construit le graphe Vamana avec l'algorithme exact de DiskANN
  - Produit le fichier _disk.index avec le layout secteur-aligné
  - C'est exactement le même code utilisé en production

CE QUE FAIT CE SCRIPT :
  1. Appelle le binaire Rust `build_disk_index` via subprocess
  2. Parse le fichier _disk.index produit pour extraire le graphe
  3. Sauvegarde les métadonnées pour l'étape d'analyse

PRÉREQUIS :
  - Rust toolchain installé (rustup)
  - Le crate diskann-tools compilé :
    cargo build --release -p diskann-tools --bin build_disk_index

USAGE :
  python 02_build_index.py --data data/nq_vectors.fbin --dim 768 --output nq_index
  python 02_build_index.py --data data/nq_vectors.fbin --dim 768 --R 64 --L 100
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from parse_disk_index import read_header, extract_graph, build_sector_to_nodes


# ─────────────────────────────────────────────────────────────────────
# Localisation du binaire Rust
# ─────────────────────────────────────────────────────────────────────

def find_rust_binary() -> str:
    """Localise le binaire build_disk_index compilé.

    POURQUOI CHERCHER À PLUSIEURS ENDROITS ?
    Le binaire peut être dans target/release/ ou target/debug/ selon
    le mode de compilation. On préfère release (plus rapide).
    """
    # Chemin du repo DiskANN (parent de experiments/)
    repo_root = Path(__file__).resolve().parent.parent.parent

    candidates = [
        repo_root / "target" / "release" / "build_disk_index",
        repo_root / "target" / "debug" / "build_disk_index",
        # Windows
        repo_root / "target" / "release" / "build_disk_index.exe",
        repo_root / "target" / "debug" / "build_disk_index.exe",
    ]

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return None


def build_rust_binary(repo_root: Path, release: bool = True) -> str:
    """Compile le binaire Rust si nécessaire.

    POURQUOI COMPILER ICI ?
    Pour que le script soit self-contained : l'utilisateur n'a pas besoin
    de connaître la commande cargo exacte.
    """
    mode_flag = "--release" if release else ""
    mode_dir = "release" if release else "debug"

    print(f"Compilation du builder Rust ({mode_dir}) ...")
    cmd = f"cargo build {mode_flag} -p diskann-tools --bin build_disk_index"
    print(f"  $ {cmd}")

    result = subprocess.run(
        cmd.split(),
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print("ERREUR de compilation :")
        print(result.stderr)
        sys.exit(1)

    binary = repo_root / "target" / mode_dir / "build_disk_index"
    if not binary.exists():
        # Windows
        binary = binary.with_suffix(".exe")
    return str(binary)


# ─────────────────────────────────────────────────────────────────────
# Construction de l'index
# ─────────────────────────────────────────────────────────────────────

def build_disk_index(
    data_path: str,
    index_prefix: str,
    dim: int,
    R: int = 64,
    L: int = 100,
    num_threads: int = 1,
    build_ram_limit_gb: float = 4.0,
    num_pq_chunks: int = 0,
    metric: str = "l2",
    compile_if_needed: bool = True,
) -> str:
    """Construit un index DiskANN via le builder Rust.

    PARAMÈTRES :
    - data_path : chemin vers le fichier .fbin (format DiskANN)
    - index_prefix : préfixe pour les fichiers de sortie
    - dim : dimensionnalité des vecteurs
    - R : degré maximum du graphe (max_degree). Plus grand = meilleur recall mais
          plus d'I/O par nœud. Valeurs typiques : 32, 64, 128.
    - L : taille de la liste de recherche pendant la construction. Plus grand =
          meilleur graphe mais construction plus lente. Typiquement L ≥ R.
    - num_threads : parallélisme de la construction
    - build_ram_limit_gb : budget RAM pour la construction
    - num_pq_chunks : nombre de chunks PQ (0 = utiliser dim comme défaut)
    - metric : "l2" ou "cosine"

    Retourne le chemin vers le fichier _disk.index produit.
    """
    repo_root = Path(__file__).resolve().parent.parent.parent

    # Trouver ou compiler le binaire
    binary = find_rust_binary()
    if binary is None:
        if compile_if_needed:
            binary = build_rust_binary(repo_root)
        else:
            print("ERREUR : le binaire build_disk_index n'est pas compilé.")
            print("  Compilez-le avec :")
            print("    cargo build --release -p diskann-tools --bin build_disk_index")
            sys.exit(1)

    # Construire la commande
    cmd = [
        binary,
        "--data-path", str(data_path),
        "--index-path-prefix", str(index_prefix),
        "--dim", str(dim),
        "--R", str(R),
        "--L", str(L),
        "--num-threads", str(num_threads),
        "--build-ram-limit-gb", str(build_ram_limit_gb),
        "--num-pq-chunks", str(num_pq_chunks),
        "--metric", metric,
    ]

    print(f"Lancement du builder DiskANN ...")
    print(f"  $ {' '.join(cmd)}")
    print()

    t0 = time.time()

    result = subprocess.run(
        cmd,
        capture_output=False,  # Laisser la sortie s'afficher en direct
        text=True,
    )

    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\nERREUR : le builder a retourné le code {result.returncode}")
        sys.exit(1)

    print(f"\nConstruction terminée en {elapsed:.1f}s")

    disk_index_file = f"{index_prefix}_disk.index"
    if not Path(disk_index_file).exists():
        print(f"ERREUR : le fichier {disk_index_file} n'a pas été produit")
        sys.exit(1)

    return disk_index_file


# ─────────────────────────────────────────────────────────────────────
# Extraction et sauvegarde des métadonnées
# ─────────────────────────────────────────────────────────────────────

def extract_and_save_metadata(
    disk_index_file: str,
    output_prefix: str,
) -> dict:
    """Parse le fichier _disk.index et sauvegarde les métadonnées.

    POURQUOI SAUVEGARDER ?
    L'étape 03 (analyse I/O) a besoin du graphe et des métadonnées.
    On les sauvegarde en JSON pour éviter de re-parser le fichier binaire.
    """
    print(f"\nExtraction du graphe depuis {disk_index_file} ...")
    t0 = time.time()

    header = read_header(disk_index_file)
    print(header.summary())
    print()

    graph, _, _ = extract_graph(disk_index_file, header, load_vectors=False)
    elapsed = time.time() - t0
    print(f"Extraction terminée en {elapsed:.1f}s")

    # Statistiques du graphe
    degrees = [len(v) for v in graph.values()]
    max_degree = max(degrees) if degrees else 0
    avg_degree = sum(degrees) / len(degrees) if degrees else 0
    print(f"  Degré max observé:   {max_degree}")
    print(f"  Degré moyen:         {avg_degree:.1f}")

    # Sauvegarder les métadonnées
    metadata = {
        "disk_index_file": str(disk_index_file),
        "num_pts": header.num_pts,
        "dims": header.dims,
        "medoid": header.medoid,
        "node_len": header.node_len,
        "max_degree": header.max_degree,
        "max_degree_observed": max_degree,
        "avg_degree": avg_degree,
        "nodes_per_sector": header.nodes_per_sector,
        "sectors_per_node": header.sectors_per_node,
        "block_size": header.block_size,
        "file_size": header.file_size,
    }

    meta_path = f"{output_prefix}_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMétadonnées sauvegardées dans {meta_path}")

    return metadata


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Construit un index DiskANN via le builder Rust"
    )
    parser.add_argument("--data", "-d", required=True,
                        help="Fichier .fbin des vecteurs (format DiskANN)")
    parser.add_argument("--dim", type=int, required=True,
                        help="Dimensionnalité des vecteurs")
    parser.add_argument("--output", "-o", default="index/nq",
                        help="Préfixe de sortie (défaut: index/nq)")
    parser.add_argument("--R", type=int, default=64,
                        help="Degré max du graphe (défaut: 64)")
    parser.add_argument("--L", type=int, default=100,
                        help="Taille liste de construction (défaut: 100)")
    parser.add_argument("--num-threads", type=int, default=1,
                        help="Nombre de threads (défaut: 1)")
    parser.add_argument("--build-ram-limit-gb", type=float, default=4.0,
                        help="Budget RAM en GB (défaut: 4.0)")
    parser.add_argument("--num-pq-chunks", type=int, default=0,
                        help="Nombre de chunks PQ (0 = dim, défaut: 0)")
    parser.add_argument("--metric", default="l2", choices=["l2", "cosine"],
                        help="Métrique de distance (défaut: l2)")
    parser.add_argument("--no-compile", action="store_true",
                        help="Ne pas compiler automatiquement le binaire Rust")
    args = parser.parse_args()

    # Créer le dossier de sortie
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Construction de l'index DiskANN")
    print("=" * 60)
    print(f"  Données:     {args.data}")
    print(f"  Dimension:   {args.dim}")
    print(f"  R:           {args.R}")
    print(f"  L:           {args.L}")
    print(f"  Métrique:    {args.metric}")
    print(f"  Sortie:      {args.output}")
    print()

    # Étape 1 : Construire l'index avec Rust
    disk_index_file = build_disk_index(
        data_path=args.data,
        index_prefix=args.output,
        dim=args.dim,
        R=args.R,
        L=args.L,
        num_threads=args.num_threads,
        build_ram_limit_gb=args.build_ram_limit_gb,
        num_pq_chunks=args.num_pq_chunks,
        metric=args.metric,
        compile_if_needed=not args.no_compile,
    )

    # Étape 2 : Extraire les métadonnées du graphe
    metadata = extract_and_save_metadata(disk_index_file, args.output)

    print()
    print("=" * 60)
    print("RÉSUMÉ")
    print("=" * 60)
    print(f"  Index disque: {disk_index_file}")
    print(f"  {metadata['num_pts']:,} points, dim={metadata['dims']}")
    print(f"  R={metadata['max_degree']}, degré moyen={metadata['avg_degree']:.1f}")
    print(f"  {metadata['nodes_per_sector']} nœud(s)/secteur")
    print(f"  Taille fichier: {metadata['file_size']:,} octets")
    print()
    print("Pour lancer l'analyse I/O :")
    print(f"  python 03_analyze_io.py --data {args.data} "
          f"--disk-index {disk_index_file} --L 100")


if __name__ == "__main__":
    main()
