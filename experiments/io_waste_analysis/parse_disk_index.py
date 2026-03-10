"""
parse_disk_index.py — Parseur du fichier _disk.index produit par DiskANN
=========================================================================

CE MODULE EST LE CŒUR DU NOUVEAU PIPELINE.

POURQUOI CE MODULE ?
--------------------
Le builder Rust de DiskANN produit un fichier `{prefix}_disk.index` qui contient :
  - Le header (métadonnées) dans le secteur 0
  - Les nœuds du graphe (vecteur + voisins) dans les secteurs suivants

Ce module lit ce fichier binaire et extrait :
  1. Les métadonnées (npoints, dim, max_degree, medoid, node_len, etc.)
  2. Le graphe de voisinage complet {node_id → [neighbor_ids]}
  3. Le mapping nœud → secteur (pour l'analyse I/O)

On n'a PAS besoin de charger les vecteurs en mémoire ici — l'analyse I/O
ne dépend que de la STRUCTURE du graphe et du LAYOUT sur disque.

FORMAT DU FICHIER (d'après le code Rust de diskann-disk/) :
============================================================

SECTEUR 0 — Header (4096 octets, zero-padded)
  Offset  0 :  u32 LE — taille en octets de la structure GraphHeader
  Offset  4 :  u32 LE — toujours 1 (nombre de "rangées" du Metadata prefix)
  Offset  8 :  u64 LE — num_pts        (nombre de vecteurs)
  Offset 16 :  u64 LE — dims           (dimensionnalité des vecteurs)
  Offset 24 :  u64 LE — medoid         (index du nœud de départ pour la recherche)
  Offset 32 :  u64 LE — node_len       (taille d'un nœud en octets)
  Offset 40 :  u64 LE — nodes_per_sector (0 si un nœud > 1 secteur)
  Offset 48 :  u64 LE — frozen_num     (nombre de points gelés)
  Offset 56 :  u64 LE — frozen_loc     (position du point gelé)
  Offset 64 :  u64 LE — reorder_data   (toujours 0, compat C++)
  Offset 72 :  u64 LE — file_size      (taille totale du fichier)
  Offset 80 :  u64 LE — assoc_data_len (taille des données associées par nœud)
  Offset 88 :  u64 LE — block_size     (taille d'un secteur, typiquement 4096)
  Offset 96 :  u32 LE — layout_major   (version du layout, typiquement 1)
  Offset 100:  u32 LE — layout_minor   (sous-version, typiquement 0)

SECTEURS 1+ — Données des nœuds
  Chaque nœud occupe exactement node_len octets :
    [0                         .. dim*sizeof(T))       : vecteur float32
    [dim*sizeof(T)             .. dim*sizeof(T)+4)     : num_neighbors (u32 LE)
    [dim*sizeof(T)+4           .. dim*sizeof(T)+4+num_neighbors*4) : neighbor_ids (u32 LE chacun)
    [après les voisins         .. node_len-assoc_data_len) : padding (zéros)
    [node_len-assoc_data_len   .. node_len)            : données associées

  Si nodes_per_sector > 0 : les nœuds sont empaquetés dans chaque secteur
    nœud i → secteur (1 + i // nodes_per_sector)
    offset dans le secteur = (i % nodes_per_sector) * node_len

  Si nodes_per_sector == 0 : un nœud s'étale sur plusieurs secteurs
    nœud i → secteur de base (1 + i * sectors_per_node)
    sectors_per_node = ceil(node_len / block_size)
"""

import struct
import math
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class DiskIndexHeader:
    """Métadonnées lues depuis le secteur 0 du fichier _disk.index."""
    num_pts: int
    dims: int
    medoid: int
    node_len: int
    nodes_per_sector: int
    frozen_num: int
    frozen_loc: int
    reorder_data: int
    file_size: int
    assoc_data_len: int
    block_size: int
    layout_major: int
    layout_minor: int

    @property
    def sectors_per_node(self) -> int:
        """Nombre de secteurs nécessaires pour un seul nœud (cas B : gros nœuds)."""
        if self.nodes_per_sector > 0:
            return 1
        return math.ceil(self.node_len / self.block_size)

    @property
    def max_degree(self) -> int:
        """Degré maximum déduit de node_len.

        Formule inverse de la construction :
          node_len = dim * sizeof(f32) + (max_degree + 1) * sizeof(u32) + assoc_data_len
          → max_degree = (node_len - dim*4 - assoc_data_len) / 4 - 1
        """
        vec_bytes = self.dims * 4  # sizeof(f32) = 4
        remaining = self.node_len - vec_bytes - self.assoc_data_len
        return (remaining // 4) - 1

    @property
    def total_data_sectors(self) -> int:
        """Nombre total de secteurs de données (excluant le header)."""
        if self.nodes_per_sector > 0:
            return math.ceil(self.num_pts / self.nodes_per_sector)
        return self.num_pts * self.sectors_per_node

    def summary(self) -> str:
        lines = [
            "=== DiskANN Disk Index Header ===",
            f"Points:             {self.num_pts:,}",
            f"Dimensions:         {self.dims}",
            f"Medoid:             {self.medoid}",
            f"Node length:        {self.node_len} bytes",
            f"Max degree (R):     {self.max_degree}",
            f"Block size:         {self.block_size} bytes",
            f"Nodes/sector:       {self.nodes_per_sector}",
            f"Sectors/node:       {self.sectors_per_node}",
            f"File size:          {self.file_size:,} bytes",
            f"Associated data:    {self.assoc_data_len} bytes",
            f"Frozen points:      {self.frozen_num}",
            f"Layout version:     {self.layout_major}.{self.layout_minor}",
            f"Total data sectors: {self.total_data_sectors:,}",
        ]
        return "\n".join(lines)


def read_header(path: str) -> DiskIndexHeader:
    """Lit le header (secteur 0) d'un fichier _disk.index.

    Le header commence par un préfixe Metadata de 8 octets :
      [taille_struct: u32][nrows: u32]
    suivi de la structure GraphHeader (80 octets) puis de champs supplémentaires.
    """
    with open(path, "rb") as f:
        # Lire le secteur 0 complet
        header_block = f.read(4096)

    if len(header_block) < 104:
        raise ValueError(f"Header trop court : {len(header_block)} octets (attendu ≥ 104)")

    # Sauter le préfixe Metadata (8 octets)
    # struct_size = u32 at offset 0
    # nrows = u32 at offset 4 (toujours 1)
    meta = header_block[8:]

    # GraphHeader : 10 champs u64 = 80 octets
    (
        num_pts,
        dims,
        medoid,
        node_len,
        nodes_per_sector,
        frozen_num,
        frozen_loc,
        reorder_data,
        file_size,
        assoc_data_len,
    ) = struct.unpack_from("<10Q", meta, 0)

    # Champs supplémentaires : block_size (u64), layout_major (u32), layout_minor (u32)
    block_size = struct.unpack_from("<Q", meta, 80)[0]
    layout_major, layout_minor = struct.unpack_from("<II", meta, 88)

    # Valider
    if block_size == 0:
        block_size = 4096  # fallback au défaut

    return DiskIndexHeader(
        num_pts=num_pts,
        dims=dims,
        medoid=medoid,
        node_len=node_len,
        nodes_per_sector=nodes_per_sector,
        frozen_num=frozen_num,
        frozen_loc=frozen_loc,
        reorder_data=reorder_data,
        file_size=file_size,
        assoc_data_len=assoc_data_len,
        block_size=block_size,
        layout_major=layout_major,
        layout_minor=layout_minor,
    )


def extract_graph(
    path: str,
    header: Optional[DiskIndexHeader] = None,
    load_vectors: bool = False,
) -> Tuple[Dict[int, List[int]], DiskIndexHeader, Optional[np.ndarray]]:
    """Extrait le graphe de voisinage depuis le fichier _disk.index.

    Retourne :
      graph : Dict[int, List[int]] — graphe {node_id → [neighbor_ids]}
      header : DiskIndexHeader — métadonnées de l'index
      vectors : Optional[np.ndarray] — vecteurs (N, dims) si load_vectors=True

    POURQUOI ON PEUT PARSER CE FORMAT ?
    Le format est déterministe : on connaît exactement la position de chaque
    nœud dans le fichier grâce au header (node_len, nodes_per_sector, block_size).
    C'est exactement ce que fait le DiskIndexReader en Rust.
    """
    if header is None:
        header = read_header(path)

    graph: Dict[int, List[int]] = {}
    vectors = np.zeros((header.num_pts, header.dims), dtype=np.float32) if load_vectors else None
    vec_bytes = header.dims * 4  # sizeof(f32)

    with open(path, "rb") as f:
        for node_id in range(header.num_pts):
            # Calculer la position du nœud dans le fichier
            if header.nodes_per_sector > 0:
                # Cas A : plusieurs nœuds par secteur
                sector_id = 1 + node_id // header.nodes_per_sector
                offset_in_sector = (node_id % header.nodes_per_sector) * header.node_len
            else:
                # Cas B : un nœud sur plusieurs secteurs
                sector_id = 1 + node_id * header.sectors_per_node
                offset_in_sector = 0

            file_offset = sector_id * header.block_size + offset_in_sector

            # Lire le nœud
            f.seek(file_offset)
            node_data = f.read(header.node_len)

            if len(node_data) < vec_bytes + 4:
                print(f"WARN: nœud {node_id} tronqué (lu {len(node_data)} octets)")
                graph[node_id] = []
                continue

            # Extraire le vecteur si demandé
            if load_vectors and vectors is not None:
                vectors[node_id] = np.frombuffer(node_data[:vec_bytes], dtype=np.float32)

            # Extraire le nombre de voisins
            num_neighbors = struct.unpack_from("<I", node_data, vec_bytes)[0]

            # Borner num_neighbors au max_degree pour éviter les corruptions
            num_neighbors = min(num_neighbors, header.max_degree)

            # Extraire les IDs des voisins
            neighbors_start = vec_bytes + 4
            neighbors_end = neighbors_start + num_neighbors * 4

            if neighbors_end <= len(node_data):
                neighbors = list(struct.unpack_from(
                    f"<{num_neighbors}I", node_data, neighbors_start
                ))
                # Filtrer les IDs invalides
                neighbors = [n for n in neighbors if n < header.num_pts]
            else:
                neighbors = []

            graph[node_id] = neighbors

    return graph, header, vectors


def get_node_sector(node_id: int, header: DiskIndexHeader) -> int:
    """Retourne le secteur contenant le nœud donné (identique à node_to_sector du layout)."""
    if header.nodes_per_sector > 0:
        return 1 + node_id // header.nodes_per_sector
    else:
        return 1 + node_id * header.sectors_per_node


def get_all_sectors_for_node(node_id: int, header: DiskIndexHeader) -> Set[int]:
    """Retourne tous les secteurs lus pour accéder au nœud donné."""
    base = get_node_sector(node_id, header)
    return set(range(base, base + header.sectors_per_node))


def build_sector_to_nodes(header: DiskIndexHeader) -> Dict[int, Set[int]]:
    """Construit le mapping secteur → ensemble de nœuds qu'il contient.

    C'est le mapping clé pour l'analyse I/O : quand on lit un secteur,
    on obtient tous ces nœuds, qu'on en ait besoin ou non.
    """
    mapping: Dict[int, Set[int]] = {}
    for nid in range(header.num_pts):
        sector = get_node_sector(nid, header)
        if sector not in mapping:
            mapping[sector] = set()
        mapping[sector].add(nid)
    return mapping


# ─────────────────────────────────────────────────────────────────────
# Main : extraction standalone
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import json
    import time

    parser = argparse.ArgumentParser(
        description="Parse un fichier DiskANN _disk.index et extrait le graphe"
    )
    parser.add_argument("index_file", help="Chemin vers le fichier _disk.index")
    parser.add_argument("--load-vectors", action="store_true",
                        help="Charger aussi les vecteurs (attention : coûteux en RAM)")
    parser.add_argument("--stats", action="store_true",
                        help="Afficher des statistiques sur le graphe")
    parser.add_argument("--output", "-o", default=None,
                        help="Sauvegarder le graphe au format JSON")
    args = parser.parse_args()

    print(f"Lecture de {args.index_file} ...")
    t0 = time.time()

    header = read_header(args.index_file)
    print(header.summary())
    print()

    graph, _, vectors = extract_graph(
        args.index_file, header, load_vectors=args.load_vectors
    )
    elapsed = time.time() - t0
    print(f"Extraction terminée en {elapsed:.1f}s")
    print(f"  {len(graph):,} nœuds extraits")

    if args.stats:
        degrees = [len(v) for v in graph.values()]
        print(f"\n=== Statistiques du graphe ===")
        print(f"  Degré max:   {max(degrees)}")
        print(f"  Degré min:   {min(degrees)}")
        print(f"  Degré moyen: {sum(degrees) / len(degrees):.1f}")
        print(f"  Degré médian:{sorted(degrees)[len(degrees)//2]}")

        # Statistiques de localité
        print(f"\n=== Localité spatiale ===")
        neighbor_same_sector = 0
        total_edges = 0
        for nid, neighbors in graph.items():
            my_sector = get_node_sector(nid, header)
            for nbr in neighbors:
                total_edges += 1
                if get_node_sector(nbr, header) == my_sector:
                    neighbor_same_sector += 1
        if total_edges > 0:
            locality = neighbor_same_sector / total_edges
            print(f"  Arêtes intra-secteur: {neighbor_same_sector:,} / {total_edges:,} "
                  f"({locality:.2%})")
            print(f"  → {1-locality:.2%} des voisins dans le graphe sont dans un "
                  f"secteur DIFFERENT du nœud courant")

    if args.output:
        print(f"\nSauvegarde dans {args.output} ...")
        # Convertir les clés en strings pour JSON
        output = {
            "header": {
                "num_pts": header.num_pts,
                "dims": header.dims,
                "medoid": header.medoid,
                "node_len": header.node_len,
                "max_degree": header.max_degree,
                "nodes_per_sector": header.nodes_per_sector,
                "block_size": header.block_size,
            },
            "graph": {str(k): v for k, v in graph.items()},
        }
        with open(args.output, "w") as f:
            json.dump(output, f)
        print("  Sauvegardé.")
