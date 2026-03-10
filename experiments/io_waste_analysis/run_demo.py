#!/usr/bin/env python3
"""
run_demo.py — Démo complète de l'analyse I/O DiskANN
======================================================

CE SCRIPT A DEUX MODES :

MODE 1 : AVEC LE BUILDER RUST (recommandé, résultats réalistes)
  python run_demo.py --use-rust --dim 128 --N 10000

  → Génère des données synthétiques au format .fbin
  → Appelle le builder Rust pour construire un vrai index DiskANN
  → Parse le fichier _disk.index pour extraire le graphe Vamana réel
  → Lance l'analyse I/O avec le vrai layout

  PRÉREQUIS : Rust toolchain + compilation du binaire :
    cargo build --release -p diskann-tools --bin build_disk_index

MODE 2 : SANS RUST (fallback, pour test rapide du pipeline)
  python run_demo.py --dim 128 --N 10000

  → Génère des données synthétiques
  → Construit un graphe kNN brute-force en Python
  → Utilise DiskLayout pour simuler le layout disque
  → Lance l'analyse I/O

  ATTENTION : le graphe kNN brute-force N'EST PAS un graphe Vamana.
  Les résultats sont indicatifs mais pas fidèles à DiskANN réel.
  Ce mode est utile uniquement pour vérifier que le code fonctionne.

USAGE :
  python run_demo.py                           # Mode fallback, config par défaut
  python run_demo.py --use-rust                # Mode Rust, config par défaut
  python run_demo.py --dim 768 --N 50000       # Simuler NQ (768-dim)
  python run_demo.py --dim 128 --R 32          # Petit degré
"""

import argparse
import json
import struct
import time
from pathlib import Path

import numpy as np

from diskann_layout import DiskLayout
from greedy_search import greedy_search


def generate_synthetic_data(N: int, dim: int, seed: int = 42) -> np.ndarray:
    """Génère des vecteurs aléatoires avec une structure en clusters.

    POURQUOI DES CLUSTERS ?
    Les vrais embeddings (BERT, DPR) ne sont pas uniformément distribués :
    ils forment des clusters thématiques. On simule ça avec un mélange
    de gaussiennes pour que le graphe kNN ait une structure réaliste.
    """
    rng = np.random.RandomState(seed)

    n_clusters = max(10, N // 1000)
    points_per_cluster = N // n_clusters
    remainder = N - points_per_cluster * n_clusters

    data = []
    for c in range(n_clusters):
        center = rng.randn(dim).astype(np.float32) * 5.0
        n_pts = points_per_cluster + (1 if c < remainder else 0)
        cluster_data = center + rng.randn(n_pts, dim).astype(np.float32) * 0.5
        data.append(cluster_data)

    return np.vstack(data).astype(np.float32)


def save_as_fbin(data: np.ndarray, path: str):
    """Sauvegarde au format DiskANN .fbin : [npoints:u32][ndims:u32][data:f32...]."""
    npoints, ndims = data.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<II", npoints, ndims))
        f.write(data.astype(np.float32).tobytes())
    print(f"  Sauvegardé {path} ({npoints:,} × {ndims})")


def build_knn_graph_bruteforce(data: np.ndarray, R: int = 64, batch_size: int = 500):
    """Construit un graphe kNN brute-force (fallback sans Rust).

    ATTENTION : ce N'EST PAS un graphe Vamana. C'est un simple kNN.
    Le graphe Vamana a des propriétés supplémentaires (navigabilité,
    pruning α, saturation) qui le rendent plus efficace pour la recherche
    mais aussi plus structuré. Utiliser le mode --use-rust pour le vrai graphe.
    """
    N = data.shape[0]
    graph = {}

    print(f"Construction du graphe kNN brute-force ({N:,} points, R={R}) ...")
    t0 = time.time()

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = data[start:end]

        batch_sq = np.sum(batch ** 2, axis=1, keepdims=True)
        data_sq = np.sum(data ** 2, axis=1, keepdims=True).T
        dists = batch_sq + data_sq - 2 * batch @ data.T

        for i in range(end - start):
            idx = start + i
            dists[i, idx] = np.inf
            neighbors = np.argsort(dists[i])[:R].tolist()
            graph[idx] = neighbors

        if start % (batch_size * 5) == 0:
            print(f"  {end:,}/{N:,}")

    elapsed = time.time() - t0
    print(f"  Terminé en {elapsed:.1f}s")

    centroid = data.mean(axis=0)
    dists_to_centroid = np.linalg.norm(data - centroid, axis=1)
    medoid = int(np.argmin(dists_to_centroid))

    return graph, medoid


# ─────────────────────────────────────────────────────────────────────
# Mode Rust : utilise le vrai builder DiskANN
# ─────────────────────────────────────────────────────────────────────

def run_demo_rust(N, dim, R, L, K, n_queries, output_dir):
    """Démo avec le vrai builder Rust de DiskANN."""
    from parse_disk_index import read_header, extract_graph, build_sector_to_nodes, get_all_sectors_for_node

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"DÉMO — Analyse I/O DiskANN (MODE RUST)")
    print(f"  N={N:,} points, dim={dim}, R={R}, L={L}, K={K}")
    print(f"  {n_queries} requêtes")
    print("=" * 60)

    # 1. Générer les données
    print("\n[1/6] Génération des données synthétiques ...")
    data = generate_synthetic_data(N, dim)
    queries = generate_synthetic_data(n_queries, dim, seed=123)
    print(f"  Données : {data.shape}, Requêtes : {queries.shape}")

    # 2. Sauvegarder au format .fbin
    print("\n[2/6] Sauvegarde au format .fbin ...")
    data_path = str(output_dir / "synthetic_data.fbin")
    save_as_fbin(data, data_path)

    # 3. Construire l'index avec Rust
    print("\n[3/6] Construction de l'index DiskANN (Rust) ...")
    # Importer la fonction build depuis notre script 02
    import importlib
    mod = importlib.import_module("02_build_index")
    build_disk_index_fn = mod.build_disk_index
    index_prefix = str(output_dir / "demo_index")
    disk_index_file = build_disk_index_fn(
        data_path=data_path,
        index_prefix=index_prefix,
        dim=dim,
        R=R,
        L=L,
        num_threads=1,
        build_ram_limit_gb=2.0,
    )

    # 4. Parser le disk index
    print("\n[4/6] Extraction du graphe depuis le disk index ...")
    header = read_header(disk_index_file)
    print(header.summary())
    graph, _, _ = extract_graph(disk_index_file, header, load_vectors=False)
    max_degree = max(len(v) for v in graph.values())
    avg_degree = sum(len(v) for v in graph.values()) / len(graph)
    print(f"  Degré max: {max_degree}, moyen: {avg_degree:.1f}")
    print(f"  Medoid: {header.medoid}")

    # 5. Analyse I/O
    print(f"\n[5/6] Analyse I/O ({n_queries} requêtes, L={L}) ...")
    sector_to_nodes = build_sector_to_nodes(header)

    run_io_analysis(data, queries, graph, header.medoid, L, K,
                    header, sector_to_nodes, n_queries, output_dir)


# ─────────────────────────────────────────────────────────────────────
# Mode fallback : graphe brute-force + layout simulé
# ─────────────────────────────────────────────────────────────────────

def run_demo_fallback(N, dim, R, L, K, n_queries, output_dir):
    """Démo avec graphe brute-force (sans Rust)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"DÉMO — Analyse I/O DiskANN (MODE FALLBACK)")
    print(f"  ⚠ Graphe kNN brute-force, PAS un vrai graphe Vamana")
    print(f"  N={N:,} points, dim={dim}, R={R}, L={L}, K={K}")
    print(f"  {n_queries} requêtes")
    print("=" * 60)

    # 1. Générer les données
    print("\n[1/5] Génération des données synthétiques ...")
    data = generate_synthetic_data(N, dim)
    queries = generate_synthetic_data(n_queries, dim, seed=123)
    print(f"  Données : {data.shape}, Requêtes : {queries.shape}")

    # 2. Construire le graphe brute-force
    print(f"\n[2/5] Construction du graphe kNN brute-force ...")
    graph, medoid = build_knn_graph_bruteforce(data, R=R)
    max_degree = max(len(v) for v in graph.values())
    print(f"  Degré max: {max_degree}, Medoid: {medoid}")

    # 3. Calculer le layout disque simulé
    print(f"\n[3/5] Calcul du layout disque ...")
    layout = DiskLayout.compute(dim=dim, max_degree=max_degree, npoints=N)
    print(layout.summary())

    # Créer un pseudo-header pour l'analyse
    from parse_disk_index import DiskIndexHeader
    header = DiskIndexHeader(
        num_pts=N,
        dims=dim,
        medoid=medoid,
        node_len=layout.node_len,
        nodes_per_sector=layout.nodes_per_sector,
        frozen_num=0,
        frozen_loc=0,
        reorder_data=0,
        file_size=0,
        assoc_data_len=0,
        block_size=4096,
        layout_major=1,
        layout_minor=0,
    )

    sector_to_nodes = layout.sector_to_nodes()

    # 4. Analyse I/O
    print(f"\n[4/5] Analyse I/O ({n_queries} requêtes, L={L}) ...")
    run_io_analysis(data, queries, graph, medoid, L, K,
                    header, sector_to_nodes, n_queries, output_dir)


# ─────────────────────────────────────────────────────────────────────
# Analyse I/O commune aux deux modes
# ─────────────────────────────────────────────────────────────────────

def run_io_analysis(data, queries, graph, medoid, L, K,
                    header, sector_to_nodes, n_queries, output_dir):
    """Exécute l'analyse I/O (commune aux modes Rust et fallback)."""
    from parse_disk_index import get_all_sectors_for_node
    from tqdm import tqdm

    per_query_results = []
    for qi in tqdm(range(n_queries), desc="Recherche + Analyse"):
        result = greedy_search(queries[qi], data, graph, medoid, L=L, K=K)
        visited = result.visited

        sectors_read = set()
        for nid in visited:
            sectors_read.update(get_all_sectors_for_node(nid, header))

        nodes_in_read_sectors = set()
        for sector in sectors_read:
            if sector in sector_to_nodes:
                nodes_in_read_sectors.update(sector_to_nodes[sector])

        useful = visited & nodes_in_read_sectors
        wasted = nodes_in_read_sectors - visited
        total = len(nodes_in_read_sectors)
        n_useful = len(useful)
        n_wasted = len(wasted)

        waste_ratio = n_wasted / max(total, 1)
        read_amp = total / max(n_useful, 1)

        good_sectors = 0
        for s in sectors_read:
            if s in sector_to_nodes:
                sn = sector_to_nodes[s]
                if len(sn & visited) >= len(sn) / 2:
                    good_sectors += 1
        locality = good_sectors / max(len(sectors_read), 1)

        per_query_results.append({
            "query_idx": qi,
            "nodes_visited": len(visited),
            "candidates_seen": len(result.candidates_seen),
            "sectors_read": len(sectors_read),
            "nodes_in_read_sectors": total,
            "useful_nodes": n_useful,
            "wasted_nodes": n_wasted,
            "waste_ratio": waste_ratio,
            "read_amplification_nodes": read_amp,
            "read_amplification_bytes": (len(sectors_read) * header.block_size) / max(n_useful * header.node_len, 1),
            "spatial_locality": locality,
            "iterations": result.iterations,
        })

    # Afficher les résultats
    print(f"\nRésultats :")
    print("=" * 65)

    waste_ratios = [q["waste_ratio"] for q in per_query_results]
    read_amps = [q["read_amplification_nodes"] for q in per_query_results]
    localities = [q["spatial_locality"] for q in per_query_results]
    nodes_vis = [q["nodes_visited"] for q in per_query_results]
    wasted_nodes = [q["wasted_nodes"] for q in per_query_results]

    print(f"\n{'Métrique':<35} {'Moyenne':>10} {'Médiane':>10} {'P95':>10}")
    print("-" * 65)
    print(f"{'Waste ratio':<35} {np.mean(waste_ratios):>9.1%} {np.median(waste_ratios):>9.1%} {np.percentile(waste_ratios, 95):>9.1%}")
    print(f"{'Read amplification (nœuds)':<35} {np.mean(read_amps):>9.2f}x {np.median(read_amps):>9.2f}x {np.percentile(read_amps, 95):>9.2f}x")
    print(f"{'Localité spatiale':<35} {np.mean(localities):>9.1%} {np.median(localities):>9.1%} {np.percentile(localities, 95):>9.1%}")
    print(f"{'Nœuds visités / requête':<35} {np.mean(nodes_vis):>9.0f} {np.median(nodes_vis):>9.0f} {np.percentile(nodes_vis, 95):>9.0f}")
    print(f"{'Nœuds gaspillés / requête':<35} {np.mean(wasted_nodes):>9.0f} {np.median(wasted_nodes):>9.0f} {np.percentile(wasted_nodes, 95):>9.0f}")

    # Interprétation
    print(f"\nINTERPRÉTATION :")
    if header.nodes_per_sector <= 1:
        print(f"  → Avec dim={header.dims} et node_len={header.node_len} octets")
        print(f"  → {header.nodes_per_sector} nœud(s) par secteur de {header.block_size} octets")
        wasted_bytes = header.block_size - header.node_len
        print(f"  → Le waste vient du padding ({wasted_bytes} octets/secteur inutilisés)")
        print(f"  → Le waste en nœuds est faible, mais le waste en OCTETS est réel")
    else:
        print(f"  → Avec dim={header.dims} et node_len={header.node_len} octets")
        print(f"  → {header.nodes_per_sector} nœuds par secteur de {header.block_size} octets")
        print(f"  → En moyenne, {np.mean(waste_ratios):.0%} des nœuds lus sont inutiles")
        print(f"  → Chaque requête lit ~{np.mean(wasted_nodes):.0f} nœuds pour rien")
        print(f"  → C'est {np.mean(read_amps):.1f}× plus de données que nécessaire")

    # Sauvegarder
    summary = {
        "layout": {
            "dim": header.dims,
            "max_degree": header.max_degree,
            "node_len": header.node_len,
            "nodes_per_sector": header.nodes_per_sector,
            "sectors_per_node": header.sectors_per_node,
            "block_size": header.block_size,
            "npoints": header.num_pts,
        },
        "num_queries": n_queries,
        "waste_ratio": {"mean": float(np.mean(waste_ratios)), "median": float(np.median(waste_ratios)), "p95": float(np.percentile(waste_ratios, 95))},
        "read_amplification_nodes": {"mean": float(np.mean(read_amps)), "median": float(np.median(read_amps))},
        "spatial_locality": {"mean": float(np.mean(localities)), "median": float(np.median(localities))},
    }

    output_dir = Path(output_dir)
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(output_dir / "per_query.json", "w") as f:
        json.dump(per_query_results, f)
    print(f"\nRésultats sauvegardés dans {output_dir}/")

    # Graphiques
    try:
        print("\nGénération des graphiques ...")
        from visualize_module import plot_all
        plot_all(summary, per_query_results, str(output_dir))
        print("Graphiques générés ✓")
    except Exception as e:
        print(f"Graphiques non générés ({e})")
        print(f"  Lancez : python 04_visualize.py --results {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Démo complète d'analyse I/O DiskANN")
    parser.add_argument("--N", type=int, default=10_000, help="Nombre de vecteurs (défaut: 10000)")
    parser.add_argument("--dim", type=int, default=128, help="Dimension (défaut: 128)")
    parser.add_argument("--R", type=int, default=64, help="Degré max du graphe (défaut: 64)")
    parser.add_argument("--L", type=int, default=100, help="Taille liste de recherche (défaut: 100)")
    parser.add_argument("--K", type=int, default=10, help="Top-K résultats (défaut: 10)")
    parser.add_argument("--queries", type=int, default=100, help="Nombre de requêtes (défaut: 100)")
    parser.add_argument("--output", default="results/demo", help="Dossier de sortie")
    parser.add_argument("--use-rust", action="store_true",
                        help="Utiliser le vrai builder Rust (recommandé, requiert compilation)")
    args = parser.parse_args()

    if args.use_rust:
        run_demo_rust(args.N, args.dim, args.R, args.L, args.K, args.queries, args.output)
    else:
        run_demo_fallback(args.N, args.dim, args.R, args.L, args.K, args.queries, args.output)


if __name__ == "__main__":
    main()
