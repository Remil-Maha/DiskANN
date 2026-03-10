"""
greedy_search.py — Implémentation fidèle du Greedy Beam Search de DiskANN
==========================================================================

FIDÉLITÉ AU CODE RUST
---------------------
Ce code reproduit exactement l'algorithme de `search_internal` dans
diskann/src/graph/index.rs et la structure `NeighborPriorityQueue` dans
diskann/src/neighbor/queue.rs :

  1. La liste de candidats est un TABLEAU TRIÉ de capacité fixe L
     (pas un heap). Quand la liste est pleine, un nouveau candidat est
     inséré en position triée et le pire est éjecté.
  2. La boucle s'arrête quand tous les nœuds dans les L premières
     positions du tableau ont été visités (marqués "expanded").
  3. Le nombre de nœuds expansés est typiquement de l'ordre de L
     (quelques centaines pour L=100), pas des centaines de milliers.

POURQUOI RÉIMPLÉMENTER EN PYTHON ?
-----------------------------------
L'objectif n'est pas la performance but l'instrumentation : on trace
exactement quels nœuds sont visités (= accès disque) pour mesurer le
gaspillage I/O et l'Overlap Ratio.
"""

import bisect
import numpy as np
from typing import Dict, List, Set, Tuple, NamedTuple
from dataclasses import dataclass, field


class SearchResult(NamedTuple):
    """Résultat d'une recherche."""
    # Top-K résultats : liste de (distance, node_id)
    top_k: List[Tuple[float, int]]
    # Ensemble de tous les nœuds visités (dont on a expansé les voisins)
    visited: Set[int]
    # Ordre de visite des nœuds (chaque visite = un accès disque)
    visited_order: List[int]
    # Ensemble de tous les nœuds considérés (ajoutés à la liste de candidats)
    candidates_seen: Set[int]
    # Nombre d'itérations (expansions de nœuds)
    iterations: int


@dataclass
class SearchStats:
    """Statistiques agrégées sur un ensemble de recherches."""
    total_queries: int = 0
    total_nodes_visited: int = 0
    total_candidates_seen: int = 0
    total_iterations: int = 0
    # Pour le calcul de la distribution
    nodes_visited_per_query: List[int] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────
# NeighborPriorityQueue — Reproduction fidèle de la structure Rust
# ─────────────────────────────────────────────────────────────────────

class NeighborPriorityQueue:
    """Reproduction de NeighborPriorityQueue de DiskANN (Rust).

    C'est un TABLEAU TRIÉ par distance croissante, de capacité fixe L.

    Comportement clé (conforme à queue.rs) :
      - insert(id, dist) : insère en position triée par distance.
        Si la liste est pleine ET dist >= distance du dernier élément,
        le candidat est REJETÉ. Sinon, on insère et on éjecte le dernier.
      - closest_notvisited() : retourne le nœud non-visité de plus petite
        distance, et le marque comme visité (cursor avance).
      - has_notvisited_node() : True si cursor < min(L, size).
    """

    def __init__(self, capacity: int):
        self.capacity = capacity      # = search_param_l = L
        self._dists: List[float] = []
        self._ids: List[int] = []
        self._visited: List[bool] = []
        self._cursor = 0              # index du prochain non-visité

    @property
    def size(self) -> int:
        return len(self._ids)

    def insert(self, node_id: int, distance: float):
        """Insère un candidat dans la liste triée.

        Exactement comme queue.rs insert() :
        - Si full et distance >= worst → reject
        - Sinon, insère en position triée, tronque à capacity
        """
        # Rejet si plein et pire que le dernier
        if self.size == self.capacity and self.size > 0:
            if distance >= self._dists[-1]:
                return

        # Position d'insertion (bisect_left pour reproduire get_lower_bound)
        pos = bisect.bisect_left(self._dists, distance)

        # Si plein, on doit d'abord éjecter le dernier
        if self.size == self.capacity:
            self._dists.pop()
            self._ids.pop()
            self._visited.pop()

        # Insérer en position
        self._dists.insert(pos, distance)
        self._ids.insert(pos, node_id)
        self._visited.insert(pos, False)

        # Mettre à jour le cursor si le nouvel élément est avant
        if pos < self._cursor:
            self._cursor = pos

    def has_notvisited_node(self) -> bool:
        """True si cursor < min(L, size) — exactement comme queue.rs."""
        return self._cursor < min(self.capacity, self.size)

    def closest_notvisited(self):
        """Retourne (dist, id) du plus proche non-visité, ou None."""
        if not self.has_notvisited_node():
            return None

        idx = self._cursor
        self._visited[idx] = True

        # Avancer le cursor au prochain non-visité
        self._cursor += 1
        while self._cursor < self.size and self._visited[self._cursor]:
            self._cursor += 1

        return (self._dists[idx], self._ids[idx])

    def get_best(self, k: int) -> List[Tuple[float, int]]:
        """Retourne les k meilleurs résultats."""
        n = min(k, self.size)
        return [(self._dists[i], self._ids[i]) for i in range(n)]

    def contains(self, node_id: int) -> bool:
        """Vérifie si un nœud est déjà dans la queue."""
        return node_id in self._ids


# ─────────────────────────────────────────────────────────────────────
# Greedy Search — Reproduction fidèle de search_internal
# ─────────────────────────────────────────────────────────────────────

def greedy_search(
    query: np.ndarray,
    data: np.ndarray,
    graph: Dict[int, List[int]],
    start_node: int,
    L: int,
    K: int = 10,
) -> SearchResult:
    """Greedy beam search fidèle à l'algorithme DiskANN (search_internal).

    ALGORITHME (conforme à index.rs search_internal) :
    --------------------------------------------------
    1. Initialiser la queue triée avec le medoid (point de départ)
    2. BOUCLE tant que has_notvisited_node() :
       a. Prendre le closest_notvisited() → c'est le nœud u à expanser
       b. Lire les voisins de u (= accès disque)
       c. Pour chaque voisin non encore vu :
          - Calculer sa distance à la requête
          - L'insérer dans la queue (rejeté si pire que le L-ième)
       d. La queue maintient automatiquement la taille ≤ L
    3. Retourner les K meilleurs de la queue

    POURQUOI L BORNE LE NOMBRE DE VISITES :
    -----------------------------------------
    La boucle s'arrête quand cursor ≥ min(L, size). Comme la queue a au
    plus L éléments, on visite AU PLUS L nœuds (souvent un peu moins car
    des insertions peuvent repousser le cursor). Typiquement ~L visites.
    """
    npoints = data.shape[0]

    # Distance L2 squared — même métrique que DiskANN par défaut
    def dist(node_id: int) -> float:
        diff = query - data[node_id]
        return float(np.dot(diff, diff))

    # Ensemble des nœuds déjà vus (pour ne pas les insérer deux fois)
    seen: Set[int] = set()

    # visited = nœuds effectivement expansés (= accès disque)
    visited: Set[int] = set()
    visited_order: List[int] = []

    # Initialiser la queue avec le point de départ
    queue = NeighborPriorityQueue(L)
    start_dist = dist(start_node)
    queue.insert(start_node, start_dist)
    seen.add(start_node)

    iterations = 0

    # Boucle principale — fidèle à search_internal
    while queue.has_notvisited_node():
        # Prendre le meilleur candidat non-visité
        best_dist, best_id = queue.closest_notvisited()

        # EXPANSION : lire les voisins de ce nœud (= accès disque)
        visited.add(best_id)
        visited_order.append(best_id)
        iterations += 1

        # Récupérer et traiter les voisins
        neighbors = graph.get(best_id, [])
        for nbr in neighbors:
            if nbr in seen or nbr >= npoints:
                continue
            seen.add(nbr)
            nbr_dist = dist(nbr)
            queue.insert(nbr, nbr_dist)

    # Extraire les K meilleurs résultats
    top_k = queue.get_best(K)

    return SearchResult(
        top_k=top_k,
        visited=visited,
        visited_order=visited_order,
        candidates_seen=seen,
        iterations=iterations,
    )


def batch_search(
    queries: np.ndarray,
    data: np.ndarray,
    graph: Dict[int, List[int]],
    start_node: int,
    L: int,
    K: int = 10,
    verbose: bool = True,
) -> Tuple[List[SearchResult], SearchStats]:
    """Lance la recherche sur un batch de requêtes.

    Retourne les résultats individuels et les statistiques agrégées.
    """
    from tqdm import tqdm

    results = []
    stats = SearchStats()

    iterator = tqdm(range(len(queries)), desc="Searching") if verbose else range(len(queries))

    for qi in iterator:
        result = greedy_search(queries[qi], data, graph, start_node, L, K)
        results.append(result)

        stats.total_queries += 1
        stats.total_nodes_visited += len(result.visited)
        stats.total_candidates_seen += len(result.candidates_seen)
        stats.total_iterations += result.iterations
        stats.nodes_visited_per_query.append(len(result.visited))

    return results, stats
