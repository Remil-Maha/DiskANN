# Analyse du gaspillage d'I/O dans DiskANN

## Question de recherche

Quand DiskANN effectue une recherche sur un index disque, il lit des **secteurs de 4 Ko**
pour accéder aux nœuds du graphe. Chaque secteur peut contenir plusieurs nœuds.

**Question** : Parmi les nœuds physiquement lus dans ces secteurs, quelle fraction est
effectivement visitée par l'algorithme de recherche, et quelle fraction est lue
inutilement (gaspillage d'I/O) ?

## Architecture du pipeline

```
.npy embeddings
      │
      ▼
┌──────────────────┐
│ 01_convert_npy_  │  Convertit .npy → .fbin (format DiskANN)
│    to_bin.py     │
└──────┬───────────┘
       │ .fbin
       ▼
┌──────────────────┐     ┌─────────────────────────────────────┐
│ 02_build_index.py│────▶│ Rust: build_disk_index (Vamana)     │
│                  │     │ diskann-tools/src/bin/               │
└──────┬───────────┘     │ → produit {prefix}_disk.index       │
       │                 └─────────────────────────────────────┘
       │ _disk.index
       ▼
┌──────────────────┐     ┌─────────────────────────────────────┐
│ 03_analyze_io.py │────▶│ parse_disk_index.py                 │
│                  │     │ → extrait le graphe + layout réel   │
└──────┬───────────┘     └─────────────────────────────────────┘
       │ results/
       ▼
┌──────────────────┐
│ 04_visualize.py  │  Graphiques : waste ratio, read amp, etc.
└──────────────────┘
```

## Prérequis

### Rust (pour construire l'index)

```bash
# Installer Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Compiler le builder d'index
cd /chemin/vers/DiskANN
cargo build --release -p diskann-tools --bin build_disk_index
```

### Python

```bash
pip install -r requirements.txt
```

## Usage rapide

### Démo (données synthétiques)

```bash
# Mode fallback (sans Rust, pour tester le pipeline)
python run_demo.py --dim 128 --N 10000

# Mode complet (avec le vrai builder Rust — recommandé)
python run_demo.py --use-rust --dim 128 --N 10000
```

### Pipeline complet (avec vos embeddings)

```bash
# 1. Convertir les embeddings .npy → .fbin
python 01_convert_npy_to_bin.py --input corpus_embeddings.npy --output data/nq_vectors.fbin

# 2. Construire l'index DiskANN (via Rust)
python 02_build_index.py --data data/nq_vectors.fbin --dim 768 --R 64 --L 100 --output index/nq

# 3. Analyser le gaspillage d'I/O
python 03_analyze_io.py --data data/nq_vectors.fbin --disk-index index/nq_disk.index --L 100

# 4. Visualiser les résultats
python 04_visualize.py --results results/
```

## Fichiers

| Fichier | Rôle |
|---------|------|
| `01_convert_npy_to_bin.py` | Conversion `.npy` → `.fbin` (format DiskANN) |
| `02_build_index.py` | Appelle le builder Rust pour construire l'index disque |
| `03_analyze_io.py` | **Script principal** : mesure le waste I/O |
| `04_visualize.py` | Génération des graphiques |
| `parse_disk_index.py` | Parseur Python du fichier `_disk.index` |
| `diskann_layout.py` | Modèle du layout disque (pour le mode fallback) |
| `greedy_search.py` | Beam search instrumenté (trace les nœuds visités) |
| `convert_utils.py` | Utilitaires de chargement `.fbin` |
| `visualize_module.py` | Module de visualisation (appelé par `04_visualize.py`) |
| `run_demo.py` | Démo complète (deux modes : Rust / fallback) |

### Binaire Rust

| Fichier | Rôle |
|---------|------|
| `diskann-tools/src/bin/build_disk_index.rs` | CLI pour construire un index disque DiskANN |

## Métriques mesurées

- **Waste ratio** : fraction de nœuds lus inutilement (= pas visités par la recherche)
- **Read amplification (nœuds)** : ratio nœuds lus / nœuds utiles
- **Read amplification (octets)** : ratio octets lus (secteurs) / octets utiles (nœuds)
- **Localité spatiale** : fraction de secteurs où ≥50% des nœuds sont utiles

## Pourquoi le builder Rust et pas hnswlib ?

hnswlib charge **tout** l'index en mémoire. Il n'y a aucune lecture disque.
L'intérêt de DiskANN est justement de faire de la recherche **sur disque**,
en chargeant les nœuds à la demande via des lectures de pages de 4 Ko.

Le builder Rust produit :
- Le **vrai graphe Vamana** (pruning α=1.2, saturation, une seule couche)
- Le **vrai layout secteur-aligné** (`_disk.index`) utilisé en production
- Le **vrai medoid** (point de départ de la recherche)

C'est ce qu'il faut pour mesurer le waste I/O réel.
