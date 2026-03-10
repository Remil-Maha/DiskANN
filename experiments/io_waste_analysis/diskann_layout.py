"""
diskann_layout.py — Modèle fidèle du layout disque DiskANN
===========================================================

POURQUOI CE MODULE ?
--------------------
DiskANN stocke les nœuds du graphe dans un fichier binaire aligné sur des
*secteurs* de 4096 octets (DISK_SECTOR_LEN dans le code Rust).
Chaque nœud contient :
  [vecteur (dim × sizeof(T))] [num_neighbors (u32)] [neighbor_ids (u32 × max_degree)] [padding]

La taille d'un nœud (node_len) est FIXE et calculée à la construction :
  node_len = dim × sizeof(T) + (max_degree + 1) × sizeof(u32)

Le +1 compte le champ « nombre de voisins » stocké devant la liste.

Selon que node_len ≤ 4096 ou non, deux cas se présentent :
  (a) Plusieurs nœuds par secteur : nodes_per_sector = 4096 // node_len
  (b) Un nœud s'étale sur ceil(node_len / 4096) secteurs

Ce module reproduit exactement cette logique pour calculer quel nœud tombe
dans quel secteur, afin de mesurer les I/O inutiles lors d'une recherche.
"""

from dataclasses import dataclass
from typing import Dict, Set
import math


# Constante identique au code Rust (diskann-disk/src/disk_index_build_parameter.rs)
DISK_SECTOR_LEN = 4096

# sizeof(f32) — DiskANN stocke les vecteurs en float32 par défaut
SIZEOF_F32 = 4

# sizeof(u32) — les IDs de voisins et le compteur sont des u32
SIZEOF_U32 = 4


@dataclass
class DiskLayout:
    """Modèle du layout disque d'un index DiskANN.

    Attributes:
        dim: Dimension des vecteurs.
        max_degree: Degré maximum du graphe (R dans la littérature).
        node_len: Taille en octets d'un nœud sur disque.
        nodes_per_sector: Nombre de nœuds par secteur (0 si un nœud > 1 secteur).
        sectors_per_node: Nombre de secteurs par nœud (1 si plusieurs nœuds/secteur).
        npoints: Nombre total de points dans l'index.
    """
    dim: int
    max_degree: int
    node_len: int
    nodes_per_sector: int
    sectors_per_node: int
    npoints: int

    @staticmethod
    def compute(dim: int, max_degree: int, npoints: int) -> "DiskLayout":
        """Calcule le layout à partir des paramètres de l'index.

        POURQUOI cette formule ?
        -------------------------
        Le code Rust (diskann-disk/src/build/builder/core.rs) calcule :
          node_len = (max_degree + 1) * sizeof(u32) + dim * sizeof(VectorDataType)
        
        Le +1 sur max_degree sert à stocker le champ `num_neighbors` (u32)
        juste avant la liste de voisins dans le même espace.
        """
        vector_size = dim * SIZEOF_F32
        neighbor_size = (max_degree + 1) * SIZEOF_U32  # +1 pour le compteur
        node_len = vector_size + neighbor_size

        if node_len <= DISK_SECTOR_LEN:
            # CAS A : plusieurs nœuds tiennent dans un secteur
            nodes_per_sector = DISK_SECTOR_LEN // node_len
            sectors_per_node = 1
        else:
            # CAS B : un nœud dépasse un secteur → on l'aligne sur le secteur suivant
            nodes_per_sector = 0
            sectors_per_node = math.ceil(node_len / DISK_SECTOR_LEN)

        return DiskLayout(
            dim=dim,
            max_degree=max_degree,
            node_len=node_len,
            nodes_per_sector=nodes_per_sector,
            sectors_per_node=sectors_per_node,
            npoints=npoints,
        )

    def node_to_sector(self, node_id: int) -> int:
        """Retourne l'index du secteur contenant le nœud `node_id`.

        POURQUOI +1 ?
        ---------------
        Le secteur 0 est réservé au header (métadonnées de l'index).
        Les données commencent au secteur 1.
        
        Formule du code Rust (disk_vertex_provider.rs, get_node_sector) :
          - Multi-nœud/secteur : sector = 1 + node_id // nodes_per_sector
          - Multi-secteur/nœud : sector = 1 + node_id * sectors_per_node
        """
        if self.nodes_per_sector > 0:
            return 1 + node_id // self.nodes_per_sector
        else:
            return 1 + node_id * self.sectors_per_node

    def sector_to_nodes(self) -> Dict[int, Set[int]]:
        """Construit le mapping inverse : secteur → ensemble de nœuds qu'il contient.

        POURQUOI ?
        ----------
        C'est le cœur de l'analyse : quand la recherche lit un secteur pour
        accéder à un nœud, TOUS les nœuds de ce secteur sont physiquement lus
        en mémoire (car le disque/OS lit par pages de 4 Ko). Ceux qui ne sont
        pas visités par la recherche représentent du gaspillage.
        """
        mapping: Dict[int, Set[int]] = {}
        for nid in range(self.npoints):
            sector = self.node_to_sector(nid)
            if sector not in mapping:
                mapping[sector] = set()
            mapping[sector].add(nid)
        return mapping

    def all_sectors_for_node(self, node_id: int) -> Set[int]:
        """Retourne TOUS les secteurs lus quand on accède au nœud node_id.

        POURQUOI ?
        ----------
        Si un nœud s'étale sur plusieurs secteurs (cas B), lire ce nœud
        nécessite de lire sectors_per_node secteurs consécutifs.
        Chaque secteur peut contenir des fragments d'autres nœuds (dans le
        cas multi-secteur, seul le nœud courant est dans ces secteurs, donc
        pas de waste au niveau nœud — mais du waste en octets de padding).
        """
        base = self.node_to_sector(node_id)
        return set(range(base, base + self.sectors_per_node))

    def wasted_space_per_sector(self) -> int:
        """Octets de padding inutilisés à la fin de chaque secteur (cas A).

        C'est l'espace résiduel : 4096 - nodes_per_sector × node_len.
        """
        if self.nodes_per_sector > 0:
            return DISK_SECTOR_LEN - self.nodes_per_sector * self.node_len
        else:
            return (self.sectors_per_node * DISK_SECTOR_LEN) - self.node_len

    def summary(self) -> str:
        """Résumé humain du layout."""
        lines = [
            f"=== DiskANN Disk Layout ===",
            f"Dimension:              {self.dim}",
            f"Max degree (R):         {self.max_degree}",
            f"Nombre de points:       {self.npoints:,}",
            f"Taille vecteur:         {self.dim * SIZEOF_F32} octets",
            f"Taille liste voisins:   {(self.max_degree + 1) * SIZEOF_U32} octets",
            f"Taille nœud (node_len): {self.node_len} octets",
            f"Taille secteur:         {DISK_SECTOR_LEN} octets",
        ]
        if self.nodes_per_sector > 0:
            lines += [
                f"Nœuds par secteur:      {self.nodes_per_sector}",
                f"Padding par secteur:    {self.wasted_space_per_sector()} octets",
                f"Nombre total secteurs:  {1 + math.ceil(self.npoints / self.nodes_per_sector)}",
            ]
        else:
            lines += [
                f"Secteurs par nœud:      {self.sectors_per_node}",
                f"Padding par nœud:       {self.wasted_space_per_sector()} octets",
                f"Nombre total secteurs:  {1 + self.npoints * self.sectors_per_node}",
            ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Auto-test rapide
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Exemple 1 : vecteurs 128-dim (SIFT), R=64
    # node_len = 128*4 + (64+1)*4 = 512 + 260 = 772 octets
    # nodes_per_sector = 4096 // 772 = 5
    layout_128 = DiskLayout.compute(dim=128, max_degree=64, npoints=100_000)
    print(layout_128.summary())
    print()

    # Exemple 2 : vecteurs 768-dim (BERT/NQ), R=64
    # node_len = 768*4 + (64+1)*4 = 3072 + 260 = 3332 octets
    # nodes_per_sector = 4096 // 3332 = 1 seul nœud par secteur !
    layout_768 = DiskLayout.compute(dim=768, max_degree=64, npoints=100_000)
    print(layout_768.summary())
    print()

    # Exemple 3 : vecteurs 768-dim, R=128 (degré plus élevé)
    # node_len = 3072 + (128+1)*4 = 3072 + 516 = 3588 octets
    # nodes_per_sector = 4096 // 3588 = 1 encore
    layout_768_r128 = DiskLayout.compute(dim=768, max_degree=128, npoints=100_000)
    print(layout_768_r128.summary())
    print()

    # Exemple 4 : vecteurs 1024-dim (grands modèles), R=64
    # node_len = 4096 + 260 = 4356 > 4096 → multi-secteur !
    layout_1024 = DiskLayout.compute(dim=1024, max_degree=64, npoints=100_000)
    print(layout_1024.summary())
