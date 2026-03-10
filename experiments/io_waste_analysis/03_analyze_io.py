#!/usr/bin/env python3
"""
03_analyze_io.py — Overlap Ratio (ξ) et gaspillage I/O dans DiskANN
=====================================================================

OBJECTIF
========
Mesurer la qualité du layout disque de DiskANN en exécutant des requêtes
réelles (NQ — Natural Questions) sur le graphe Vamana et en instrumentant
chaque accès disque.

MÉTRIQUE PRINCIPALE : Overlap Ratio OR(B(u))
=============================================
Pour chaque bloc chargé depuis le disque lors de la recherche, on calcule
la proportion de nœuds co-localisés qui sont des voisins du nœud cible :

    OR(B(u)) = |B(u) ∩ N(u)| / (|B(u)| - 1)    si |B(u)| > 1

où :
  - u     = le nœud cible (celui pour lequel on a déclenché la lecture)
  - B(u)  = ensemble des nœuds dans le bloc/secteur contenant u
  - N(u)  = ensemble des voisins de u dans le graphe Vamana
  - |B(u)| - 1 = nombre de nœuds AUTRES que u dans le même bloc

Note : u ∉ N(u) (pas d'auto-boucle dans Vamana), donc
B(u) ∩ N(u) = (B(u) \ {u}) ∩ N(u).

Le ξ global est la moyenne de OR(B(u)) sur TOUS les blocs chargés
pendant l'exécution de TOUTES les requêtes.

INTERPRÉTATION
==============
  ξ ≈ 0 → Les nœuds co-localisés sur disque ne sont PAS des voisins
           dans le graphe → le layout séquentiel est indépendant du graphe
  ξ ≈ 1 → Les nœuds co-localisés SONT des voisins → layout optimal

Pour dim=384, R=64, nodes_per_sector=2 :
  → Chaque secteur contient exactement 2 nœuds (u et un autre)
  → OR ∈ {0, 1} : binaire (le voisin de secteur est-il un voisin du graphe ?)
  → ξ attendu ≈ R/N ≈ 64/2.7M ≈ 0.00002 si layout aléatoire

PROTOCOLE EXPÉRIMENTAL
======================
Pour chaque requête NQ exécutée :
  1. Exécuter le beam search greedy (identique à iterate_to_fixed_point)
  2. Pour chaque nœud u visité (= chaque accès disque) :
     a. Identifier le secteur chargé et tous les nœuds B(u) qu'il contient
     b. Récupérer la liste d'adjacence N(u) du graphe Vamana
     c. Calculer OR(B(u)) = |B(u) ∩ N(u)| / (|B(u)| - 1)
  3. Calculer ξ_query = moyenne des OR sur les blocs de cette requête

À la fin, ξ_global = moyenne sur tous les blocs de toutes les requêtes.

MÉTRIQUES ADDITIONNELLES
=========================
  - Useful Nodes : nœuds chargés qui sont réellement visités par la recherche
  - Waste Ratio  : fraction de nœuds chargés non visités (= gaspillage effectif)
  - Read Amplification : facteur de sur-lecture (nœuds chargés / nœuds visités)

USAGE
=====
  python 03_analyze_io.py \\
      --data data/corpus_vectors.fbin \\
      --disk-index index/nq_disk.index \\
      --queries data/query_vectors.fbin \\
      --L 100
"""

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from tqdm import tqdm

from convert_utils import load_diskann_bin
from parse_disk_index import (
    DiskIndexHeader,
    read_header,
    extract_graph,
    get_node_sector,
    build_sector_to_nodes,
)
from greedy_search import greedy_search


# ─────────────────────────────────────────────────────────────────────
# Structures de données pour l'analyse bloc par bloc
# ─────────────────────────────────────────────────────────────────────

@dataclass
class BlockAccess:
    """Enregistrement d'un accès disque individuel.

    Quand le beam search visite le nœud u (= expanse ses voisins), cela
    déclenche la lecture du secteur contenant u. Ce record capture :
      - Quel nœud a déclenché la lecture (target_node = u)
      - Quel secteur a été chargé (sector_id)
      - Combien de nœuds sont dans ce secteur (block_size = |B(u)|)
      - Combien de ces nœuds sont des voisins de u (n_neighbors_in_block)
      - L'Overlap Ratio OR(B(u))
      - Combien de nœuds co-localisés sont aussi visités par la requête
    """
    target_node: int          # u : le nœud qui a déclenché la lecture
    sector_id: int            # identifiant du secteur chargé
    block_size: int           # |B(u)| : nombre de nœuds dans le secteur
    n_neighbors_in_block: int # |B(u) ∩ N(u)| : voisins de u co-localisés
    overlap_ratio: float      # OR(B(u)) = |B(u)∩N(u)| / (|B(u)|-1), NaN si |B(u)|=1
    n_visited_in_block: int   # |B(u) ∩ visited| - 1 : co-localisés aussi visités (excl. u)


@dataclass
class QueryAnalysis:
    """Résultat de l'analyse I/O pour une seule requête NQ."""
    query_idx: int

    # ── Overlap Ratio ──
    xi: float                      # ξ pour cette requête = mean(OR) sur les blocs
    or_values: List[float]         # OR(B(u)) pour chaque bloc où |B(u)| > 1
    block_accesses: List[BlockAccess]  # détail de chaque accès disque

    # ── Utilisation effective des données chargées ──
    nodes_visited: int             # nœuds expansés par la recherche
    nodes_loaded: int              # total nœuds dans les secteurs lus
    useful_nodes: int              # nœuds chargés ET visités
    wasted_nodes: int              # nœuds chargés mais NON visités

    # ── Ratios ──
    waste_ratio: float             # wasted / loaded
    useful_ratio: float            # useful / loaded
    read_amplification: float      # loaded / visited

    # ── Secteurs ──
    sectors_read: int              # nombre de secteurs distincts lus


# ─────────────────────────────────────────────────────────────────────
# Cœur de l'analyse : calcul de OR(B(u)) pour chaque accès disque
# ─────────────────────────────────────────────────────────────────────

def analyze_single_query(
    query_idx: int,
    search_result,
    graph: Dict[int, List[int]],
    header: DiskIndexHeader,
    sector_to_nodes: Dict[int, Set[int]],
) -> QueryAnalysis:
    """Analyse I/O complète pour une requête avec calcul de l'Overlap Ratio.

    Pour chaque nœud u visité pendant la recherche :
    ─────────────────────────────────────────────────
    1. Identifier le secteur S contenant u
    2. B(u) = sector_to_nodes[S]  (tous les nœuds dans ce secteur)
    3. N(u) = graph[u]            (voisins de u dans le graphe Vamana)
    4. Si |B(u)| > 1 :
         OR(B(u)) = |B(u) ∩ N(u)| / (|B(u)| - 1)
       Sinon :
         bloc à nœud unique → OR non défini (NaN)

    Le ξ de cette requête = moyenne des OR sur tous les blocs avec |B(u)| > 1.
    """
    visited = search_result.visited

    block_accesses: List[BlockAccess] = []
    or_values: List[float] = []

    # Pour les métriques classiques (waste ratio)
    all_loaded_nodes: Set[int] = set()
    sectors_read_set: Set[int] = set()

    # ── Pour chaque nœud visité = chaque accès disque ──
    # On utilise visited_order pour itérer dans l'ordre réel d'expansion
    for u in search_result.visited_order:
        sector = get_node_sector(u, header)
        sectors_read_set.add(sector)

        # B(u) : tous les nœuds dans le secteur contenant u
        B_u = sector_to_nodes.get(sector, {u})
        all_loaded_nodes.update(B_u)

        # N(u) : voisins de u dans le graphe Vamana
        N_u = set(graph.get(u, []))

        # Nœuds AUTRES que u dans le même bloc
        B_u_others = B_u - {u}

        # Overlap : nœuds co-localisés ET voisins de u dans le graphe
        overlap = B_u_others & N_u
        n_neighbors_in_block = len(overlap)

        # OR(B(u)) = |B(u) ∩ N(u)| / (|B(u)| - 1)
        if len(B_u) > 1:
            or_val = n_neighbors_in_block / (len(B_u) - 1)
            or_values.append(or_val)
        else:
            or_val = float('nan')

        # Combien de nœuds co-localisés sont aussi visités par cette requête ?
        # (= nœuds utiles parmi les données chargées "gratuitement")
        n_visited_in_block = len(B_u_others & visited)

        block_accesses.append(BlockAccess(
            target_node=u,
            sector_id=sector,
            block_size=len(B_u),
            n_neighbors_in_block=n_neighbors_in_block,
            overlap_ratio=or_val,
            n_visited_in_block=n_visited_in_block,
        ))

    # ── ξ pour cette requête ──
    xi = float(np.mean(or_values)) if or_values else 0.0

    # ── Métriques classiques : utilisation effective ──
    useful = len(all_loaded_nodes & visited)
    loaded = len(all_loaded_nodes)
    wasted = loaded - useful

    return QueryAnalysis(
        query_idx=query_idx,
        xi=xi,
        or_values=or_values,
        block_accesses=block_accesses,
        nodes_visited=len(visited),
        nodes_loaded=loaded,
        useful_nodes=useful,
        wasted_nodes=wasted,
        waste_ratio=wasted / max(loaded, 1),
        useful_ratio=useful / max(loaded, 1),
        read_amplification=loaded / max(len(visited), 1),
        sectors_read=len(sectors_read_set),
    )


# ─────────────────────────────────────────────────────────────────────
# Exécution sur un batch de requêtes
# ─────────────────────────────────────────────────────────────────────

def run_analysis(
    data: np.ndarray,
    queries: np.ndarray,
    graph: Dict[int, List[int]],
    header: DiskIndexHeader,
    L: int,
    K: int = 10,
) -> Tuple[Dict, List[QueryAnalysis]]:
    """Exécute l'analyse complète sur toutes les requêtes NQ.

    Pour chaque requête :
      1. Beam search greedy sur le graphe Vamana (avec medoid réel)
      2. Instrumentation des accès disque
      3. Calcul de OR(B(u)) pour chaque bloc chargé

    Retourne :
      summary : Dict avec ξ global et statistiques agrégées
      per_query : List[QueryAnalysis] avec les détails par requête
    """
    print(header.summary())
    print()

    # Pré-calcul du mapping secteur → nœuds
    print("Pré-calcul du mapping secteur → nœuds ...")
    sector_to_nodes = build_sector_to_nodes(header)
    print(f"  {len(sector_to_nodes):,} secteurs de données")

    nodes_per_sector_counts = [len(v) for v in sector_to_nodes.values()]
    print(f"  Nœuds par secteur : min={min(nodes_per_sector_counts)}, "
          f"max={max(nodes_per_sector_counts)}, "
          f"moy={np.mean(nodes_per_sector_counts):.1f}")
    print()

    n_queries = len(queries)
    print(f"Exécution de {n_queries:,} requêtes NQ réelles (L={L}, K={K}) ...")
    print(f"  Medoid (point de départ) : {header.medoid}")
    print()

    per_query_analyses: List[QueryAnalysis] = []

    for qi in tqdm(range(n_queries), desc="Requêtes NQ"):
        # Beam search greedy sur le graphe Vamana
        search_result = greedy_search(
            query=queries[qi],
            data=data,
            graph=graph,
            start_node=header.medoid,
            L=L,
            K=K,
        )

        # Analyse I/O avec calcul de OR pour chaque bloc
        analysis = analyze_single_query(
            query_idx=qi,
            search_result=search_result,
            graph=graph,
            header=header,
            sector_to_nodes=sector_to_nodes,
        )
        per_query_analyses.append(analysis)

    # Agréger
    summary = aggregate_results(per_query_analyses, header)
    return summary, per_query_analyses


# ─────────────────────────────────────────────────────────────────────
# Agrégation des résultats
# ─────────────────────────────────────────────────────────────────────

def aggregate_results(
    per_query: List[QueryAnalysis],
    header: DiskIndexHeader,
) -> Dict:
    """Calcule les statistiques globales.

    MÉTRIQUE CLÉ :
      ξ_global = moyenne de TOUS les OR(B(u)) sur TOUS les blocs chargés
                 pour TOUTES les requêtes.

    Ce n'est PAS la moyenne des ξ par requête (ce qui donnerait un poids
    égal à chaque requête indépendamment du nombre de blocs). C'est la
    moyenne pondérée naturelle : chaque accès disque compte autant.
    """

    def stats(values: List[float]) -> Dict:
        if not values:
            return {
                "mean": 0.0, "median": 0.0, "std": 0.0,
                "min": 0.0, "max": 0.0,
                "p5": 0.0, "p25": 0.0, "p75": 0.0, "p95": 0.0,
            }
        arr = np.array(values)
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "p5": float(np.percentile(arr, 5)),
            "p25": float(np.percentile(arr, 25)),
            "p75": float(np.percentile(arr, 75)),
            "p95": float(np.percentile(arr, 95)),
        }

    # ── ξ global : moyenne de TOUS les OR de TOUS les blocs ──
    all_or_values: List[float] = []
    for qa in per_query:
        all_or_values.extend(qa.or_values)
    xi_global = float(np.mean(all_or_values)) if all_or_values else 0.0

    # ── ξ par requête : distribution ──
    xi_per_query = [qa.xi for qa in per_query]

    # ── Métriques classiques ──
    waste_ratios = [qa.waste_ratio for qa in per_query]
    useful_ratios = [qa.useful_ratio for qa in per_query]
    read_amps = [qa.read_amplification for qa in per_query]
    nodes_visited = [qa.nodes_visited for qa in per_query]
    nodes_loaded = [qa.nodes_loaded for qa in per_query]
    sectors_read = [qa.sectors_read for qa in per_query]
    wasted = [qa.wasted_nodes for qa in per_query]

    # ── Compteurs globaux ──
    total_block_accesses = sum(len(qa.block_accesses) for qa in per_query)
    total_blocks_with_or = len(all_or_values)
    total_blocks_single_node = total_block_accesses - total_blocks_with_or

    # ── Distribution de OR : combien de blocs avec OR=0 vs OR>0 ──
    or_zero = sum(1 for v in all_or_values if v == 0.0)
    or_nonzero = sum(1 for v in all_or_values if v > 0.0)

    return {
        "layout": {
            "dim": header.dims,
            "max_degree": header.max_degree,
            "node_len": header.node_len,
            "nodes_per_sector": header.nodes_per_sector,
            "sectors_per_node": header.sectors_per_node,
            "block_size": header.block_size,
            "npoints": header.num_pts,
            "medoid": header.medoid,
        },
        "num_queries": len(per_query),

        # ── MÉTRIQUE PRINCIPALE : ξ (Overlap Ratio) ──
        "xi_global": xi_global,
        "xi_per_query": stats(xi_per_query),
        "or_all_blocks": stats(all_or_values),
        "or_distribution": {
            "total_block_accesses": total_block_accesses,
            "blocks_with_or_computed": total_blocks_with_or,
            "blocks_single_node_skipped": total_blocks_single_node,
            "blocks_or_zero": or_zero,
            "blocks_or_nonzero": or_nonzero,
            "fraction_or_zero": or_zero / max(total_blocks_with_or, 1),
            "fraction_or_nonzero": or_nonzero / max(total_blocks_with_or, 1),
        },

        # ── Métriques classiques ──
        "waste_ratio": stats(waste_ratios),
        "useful_ratio": stats(useful_ratios),
        "read_amplification": stats(read_amps),
        "nodes_visited_per_query": stats(nodes_visited),
        "nodes_loaded_per_query": stats(nodes_loaded),
        "sectors_read_per_query": stats(sectors_read),
        "wasted_nodes_per_query": stats(wasted),
    }


# ─────────────────────────────────────────────────────────────────────
# Détection automatique des requêtes NQ
# ─────────────────────────────────────────────────────────────────────

def find_query_file(data_dir: Path) -> Optional[Path]:
    """Cherche automatiquement un fichier de requêtes NQ dans le dossier data."""
    candidates = [
        data_dir / "query_vectors.fbin",
        data_dir / "query_embeddings.npy",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


# ─────────────────────────────────────────────────────────────────────
# Point d'entrée principal
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Mesure de l'Overlap Ratio (ξ) et du gaspillage I/O dans DiskANN"
    )
    parser.add_argument(
        "--data", "-d", required=True,
        help="Fichier .fbin des vecteurs du corpus",
    )
    parser.add_argument(
        "--disk-index", "-i", required=True,
        help="Fichier _disk.index produit par le builder Rust",
    )
    parser.add_argument(
        "--queries", "-q", default=None,
        help="Fichier de requêtes NQ (.fbin ou .npy). Auto-détecté si absent.",
    )
    parser.add_argument(
        "--L", type=int, default=100,
        help="Taille liste de recherche (défaut: 100)",
    )
    parser.add_argument(
        "--K", type=int, default=10,
        help="Nombre de résultats (défaut: 10)",
    )
    parser.add_argument(
        "--max-queries", type=int, default=None,
        help="Limiter le nombre de requêtes (pour test rapide)",
    )
    parser.add_argument(
        "--output", "-o", default="results",
        help="Dossier de sortie (défaut: results/)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  ANALYSE DE L'OVERLAP RATIO (ξ) — DiskANN Disk Index")
    print("  Requêtes NQ réelles — Graphe Vamana réel — Layout disque réel")
    print("=" * 70)
    print()

    # ─── 1. Lire le header du disk index ───
    print(f"1. Lecture du header : {args.disk_index}")
    header = read_header(args.disk_index)
    print(header.summary())
    print()

    # ─── 2. Extraire le graphe ET les vecteurs depuis le disk index ───
    print("2. Extraction du graphe et des vecteurs depuis _disk.index ...")
    t0 = time.time()
    graph, _, vectors = extract_graph(args.disk_index, header, load_vectors=True)
    elapsed = time.time() - t0
    print(f"   {len(graph):,} nœuds extraits en {elapsed:.1f}s")

    degrees = [len(v) for v in graph.values()]
    print(f"   Degré max observé : {max(degrees)}")
    print(f"   Degré moyen :       {np.mean(degrees):.1f}")
    print()

    # Utiliser les vecteurs du disk index pour garantir la cohérence
    if vectors is not None:
        data = vectors
        print(f"   Vecteurs : {data.shape[0]:,} × {data.shape[1]}d (depuis disk index)")
    else:
        print(f"   Chargement des vecteurs depuis {args.data} ...")
        data = load_diskann_bin(args.data)
        print(f"   Vecteurs : {data.shape[0]:,} × {data.shape[1]}d")
    print()

    # ─── 3. Charger les requêtes NQ réelles ───
    print("3. Chargement des requêtes NQ réelles ...")
    if args.queries:
        query_path = Path(args.queries)
    else:
        data_dir = Path(args.data).parent
        query_path = find_query_file(data_dir)
        if query_path:
            print(f"   Auto-détecté : {query_path}")
        else:
            print("   ERREUR : Aucun fichier de requêtes trouvé !")
            print("   Fournissez --queries data/query_vectors.fbin")
            print("   ou placez query_vectors.fbin dans le dossier data/")
            return

    if str(query_path).endswith(".npy"):
        queries = np.load(str(query_path)).astype(np.float32)
    else:
        queries = load_diskann_bin(str(query_path))

    if args.max_queries:
        queries = queries[:args.max_queries]
    print(f"   {queries.shape[0]:,} requêtes NQ × {queries.shape[1]}d")
    print()

    # ─── 4. Lancer l'analyse ───
    print("4. Analyse I/O avec calcul de l'Overlap Ratio OR(B(u))")
    print("   Formule : OR(B(u)) = |B(u) ∩ N(u)| / (|B(u)| - 1)")
    print("   B(u) = nœuds dans le secteur de u, N(u) = voisins de u")
    print("─" * 70)
    t0 = time.time()
    summary, per_query = run_analysis(
        data=data,
        queries=queries,
        graph=graph,
        header=header,
        L=args.L,
        K=args.K,
    )
    elapsed = time.time() - t0

    # ─── 5. Afficher les résultats ───
    print()
    print("=" * 70)
    print("  RÉSULTATS")
    print("=" * 70)
    print(f"  Temps total    : {elapsed:.1f}s ({elapsed / len(queries) * 1000:.1f} ms/requête)")
    print(f"  Requêtes NQ    : {len(queries):,}")
    print(f"  Paramètres     : L={args.L}, K={args.K}")
    print()

    # ── ξ (Overlap Ratio) ──
    xi = summary["xi_global"]
    xi_pq = summary["xi_per_query"]
    or_all = summary["or_all_blocks"]
    or_dist = summary["or_distribution"]

    print("─" * 70)
    print("  ξ (OVERLAP RATIO) — Localité graphe–disque")
    print("─" * 70)
    print(f"  ξ global             = {xi:.6f}")
    print(f"  ξ par requête        : moy={xi_pq['mean']:.6f}  méd={xi_pq['median']:.6f}  σ={xi_pq['std']:.6f}")
    print(f"  OR par bloc          : moy={or_all['mean']:.6f}  méd={or_all['median']:.6f}")
    print(f"                         P5={or_all['p5']:.6f}  P95={or_all['p95']:.6f}")
    print()
    print(f"  Accès disque totaux  : {or_dist['total_block_accesses']:,}")
    print(f"  Blocs multi-nœuds   : {or_dist['blocks_with_or_computed']:,} (OR calculé)")
    print(f"  Blocs mono-nœud     : {or_dist['blocks_single_node_skipped']:,} (OR non applicable)")
    print(f"  Blocs avec OR = 0   : {or_dist['blocks_or_zero']:,} ({or_dist['fraction_or_zero']:.2%})")
    print(f"  Blocs avec OR > 0   : {or_dist['blocks_or_nonzero']:,} ({or_dist['fraction_or_nonzero']:.2%})")
    print()

    # ── Utilisation effective ──
    wr = summary["waste_ratio"]
    ur = summary["useful_ratio"]

    print("─" * 70)
    print("  UTILISATION DES NŒUDS CHARGÉS")
    print("─" * 70)
    print(f"  Nœuds utiles (visités)   : {ur['mean']:.2%} (moy)  {ur['median']:.2%} (méd)")
    print(f"  Nœuds gaspillés          : {wr['mean']:.2%} (moy)  {wr['median']:.2%} (méd)")
    print()

    # ── Read amplification ──
    ra = summary["read_amplification"]
    nv = summary["nodes_visited_per_query"]
    nl = summary["nodes_loaded_per_query"]
    sr = summary["sectors_read_per_query"]

    print("─" * 70)
    print("  AMPLIFICATION DE LECTURE")
    print("─" * 70)
    print(f"  Read amplification       : {ra['mean']:.2f}x (moy)  {ra['median']:.2f}x (méd)")
    print(f"  Nœuds visités/requête    : {nv['mean']:.0f} (moy)  {nv['median']:.0f} (méd)")
    print(f"  Nœuds chargés/requête    : {nl['mean']:.0f} (moy)  {nl['median']:.0f} (méd)")
    print(f"  Secteurs lus/requête     : {sr['mean']:.0f} (moy)  {sr['median']:.0f} (méd)")
    print()

    # ── Interprétation ──
    print("─" * 70)
    print("  INTERPRÉTATION")
    print("─" * 70)
    nps = header.nodes_per_sector
    expected_random = header.max_degree / max(header.num_pts - 1, 1)
    print(f"  Nodes/secteur            : {nps}")
    print(f"  ξ attendu (layout aléatoire) ≈ R/(N-1) = "
          f"{header.max_degree}/({header.num_pts - 1:,}) ≈ {expected_random:.6f}")
    print(f"  ξ mesuré                 = {xi:.6f}")
    ratio = xi / expected_random if expected_random > 0 else float('inf')
    print(f"  Ratio ξ_mesuré / ξ_aléatoire = {ratio:.2f}x")
    print()
    if ratio < 2:
        print("  → Le layout disque est quasi ALÉATOIRE par rapport au graphe Vamana.")
        print("    Les nœuds co-localisés ne sont pas des voisins dans le graphe.")
        print("    → Gaspillage I/O confirmé : chaque lecture de secteur charge des nœuds inutiles.")
    elif ratio < 10:
        print("  → Légère corrélation entre layout disque et graphe, mais faible.")
    else:
        print("  → Forte corrélation entre layout disque et graphe.")
    print()

    # ─── 6. Sauvegarder les résultats ───
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Summary JSON
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Résumé           → {summary_path}")

    # Per-query JSON (compact, sans les block_accesses pour la taille)
    per_query_data = []
    for qa in per_query:
        pq = {
            "query_idx": qa.query_idx,
            "xi": round(qa.xi, 6),
            "nodes_visited": qa.nodes_visited,
            "nodes_loaded": qa.nodes_loaded,
            "useful_nodes": qa.useful_nodes,
            "wasted_nodes": qa.wasted_nodes,
            "waste_ratio": round(qa.waste_ratio, 6),
            "useful_ratio": round(qa.useful_ratio, 6),
            "read_amplification": round(qa.read_amplification, 4),
            "sectors_read": qa.sectors_read,
            "n_block_accesses": len(qa.block_accesses),
            "or_values": [round(v, 6) for v in qa.or_values],
        }
        per_query_data.append(pq)

    per_query_path = output_dir / "per_query.json"
    with open(per_query_path, "w") as f:
        json.dump(per_query_data, f, indent=1)
    print(f"  Détails/requête  → {per_query_path}")

    # Block accesses détaillés (JSONL : un record par ligne, compact)
    block_path = output_dir / "block_accesses.jsonl"
    n_records = 0
    with open(block_path, "w") as f:
        for qa in per_query:
            for ba in qa.block_accesses:
                record = {
                    "q": qa.query_idx,
                    "u": ba.target_node,
                    "sector": ba.sector_id,
                    "B_size": ba.block_size,
                    "neighbors_in_B": ba.n_neighbors_in_block,
                    "OR": round(ba.overlap_ratio, 6) if not math.isnan(ba.overlap_ratio) else None,
                    "visited_in_B": ba.n_visited_in_block,
                }
                f.write(json.dumps(record) + "\n")
                n_records += 1
    print(f"  Accès blocs      → {block_path} ({n_records:,} records)")
    print()
    print("Pour visualiser : python 04_visualize.py --results", output_dir)


if __name__ == "__main__":
    main()
