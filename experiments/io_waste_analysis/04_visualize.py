#!/usr/bin/env python3
"""
04_visualize.py — Graphiques d'analyse du gaspillage d'I/O
============================================================

POURQUOI CES GRAPHIQUES ?
--------------------------
Pour convaincre dans un paper/présentation que le layout séquentiel de DiskANN
cause un gaspillage d'I/O significatif, on produit :

1. HISTOGRAMME du waste ratio par requête
   → Montre la distribution : la plupart des requêtes gaspillent beaucoup
   
2. SCATTER PLOT : nœuds visités vs nœuds gaspillés
   → Montre la corrélation linéaire : plus on visite, plus on gaspille
   
3. BAR CHART comparant différents scénarios de dimensions/degré
   → Montre l'impact des paramètres sur le waste

4. BOX PLOT de la read amplification
   → Montre la variabilité entre requêtes

5. HEATMAP de la localité spatiale
   → Montre visuellement quels secteurs sont "utiles" vs "gaspillés"
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Backend non-interactif pour serveurs sans display
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np


def load_results(results_dir: str):
    """Charge les résultats depuis summary.json et per_query.json."""
    summary_path = Path(results_dir) / "summary.json"
    per_query_path = Path(results_dir) / "per_query.json"

    with open(summary_path) as f:
        summary = json.load(f)
    with open(per_query_path) as f:
        per_query = json.load(f)

    return summary, per_query


def plot_waste_ratio_histogram(per_query, output_dir):
    """Histogramme de la distribution du waste ratio.

    POURQUOI ?
    Le waste ratio moyen seul ne dit pas tout. La distribution montre
    si le problème est systématique (pic serré à haut waste) ou
    variable (distribution étalée).
    """
    waste_ratios = [q["waste_ratio"] for q in per_query]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(waste_ratios, bins=50, color="#e74c3c", alpha=0.8, edgecolor="white")
    ax.axvline(np.mean(waste_ratios), color="black", linestyle="--", linewidth=2,
               label=f"Moyenne: {np.mean(waste_ratios):.1%}")
    ax.axvline(np.median(waste_ratios), color="blue", linestyle=":", linewidth=2,
               label=f"Médiane: {np.median(waste_ratios):.1%}")

    ax.set_xlabel("Waste Ratio (fraction de nœuds lus inutilement)", fontsize=12)
    ax.set_ylabel("Nombre de requêtes", fontsize=12)
    ax.set_title("Distribution du gaspillage d'I/O par requête", fontsize=14)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    path = Path(output_dir) / "waste_ratio_histogram.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✓ {path}")


def plot_visited_vs_wasted(per_query, output_dir):
    """Scatter plot : nœuds visités vs nœuds gaspillés.

    POURQUOI ?
    Montre que le gaspillage croît proportionnellement avec le nombre
    de nœuds visités. La pente de cette relation est directement liée
    au nombre de nœuds par secteur.
    """
    visited = [q["nodes_visited"] for q in per_query]
    wasted = [q["wasted_nodes"] for q in per_query]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(visited, wasted, alpha=0.4, s=10, color="#3498db")

    # Ligne de tendance
    z = np.polyfit(visited, wasted, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(visited), max(visited), 100)
    ax.plot(x_line, p(x_line), "r--", linewidth=2,
            label=f"Tendance: {z[0]:.1f}x + {z[1]:.0f}")

    ax.set_xlabel("Nœuds visités par requête", fontsize=12)
    ax.set_ylabel("Nœuds lus inutilement", fontsize=12)
    ax.set_title("Corrélation entre activité de recherche et gaspillage I/O", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    path = Path(output_dir) / "visited_vs_wasted.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✓ {path}")


def plot_read_amplification_boxplot(per_query, output_dir):
    """Box plot de la read amplification (en nœuds et en octets).

    POURQUOI ?
    Le box plot montre clairement la médiane, les quartiles et les outliers.
    C'est le format préféré dans les papers pour comparer des distributions.
    """
    ra_nodes = [q["read_amplification_nodes"] for q in per_query]
    ra_bytes = [q["read_amplification_bytes"] for q in per_query]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    bp1 = axes[0].boxplot(ra_nodes, vert=True, patch_artist=True)
    bp1["boxes"][0].set_facecolor("#e74c3c")
    axes[0].set_ylabel("Read Amplification (×)", fontsize=12)
    axes[0].set_title("En nombre de nœuds", fontsize=13)
    axes[0].set_xticklabels(["Nœuds"])
    axes[0].grid(axis="y", alpha=0.3)

    bp2 = axes[1].boxplot(ra_bytes, vert=True, patch_artist=True)
    bp2["boxes"][0].set_facecolor("#3498db")
    axes[1].set_ylabel("Read Amplification (×)", fontsize=12)
    axes[1].set_title("En octets", fontsize=13)
    axes[1].set_xticklabels(["Octets"])
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle("Read Amplification dans DiskANN", fontsize=14, y=1.02)

    path = Path(output_dir) / "read_amplification_boxplot.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")


def plot_sectors_vs_useful(per_query, output_dir):
    """Ratio de secteurs avec bonne localité.

    POURQUOI ?
    Montre que la majorité des secteurs lus contiennent très peu de nœuds
    utiles — preuve directe de la mauvaise co-localisation.
    """
    localities = [q["spatial_locality"] for q in per_query]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(localities, bins=30, color="#2ecc71", alpha=0.8, edgecolor="white")
    ax.axvline(np.mean(localities), color="black", linestyle="--", linewidth=2,
               label=f"Moyenne: {np.mean(localities):.1%}")

    ax.set_xlabel("Localité spatiale (fraction de secteurs avec ≥50% utiles)", fontsize=12)
    ax.set_ylabel("Nombre de requêtes", fontsize=12)
    ax.set_title("Qualité de la co-localisation données/graphe", fontsize=14)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    path = Path(output_dir) / "spatial_locality_histogram.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✓ {path}")


def plot_summary_table(summary, output_dir):
    """Tableau récapitulatif sous forme d'image.

    POURQUOI ?
    Un tableau visuel est facile à insérer dans un paper ou une présentation.
    """
    layout = summary["layout"]
    wr = summary["waste_ratio"]
    ra = summary["read_amplification_nodes"]

    rows = [
        ["Dimension", str(layout["dim"])],
        ["Max degree (R)", str(layout["max_degree"])],
        ["Nombre de points", f"{layout['npoints']:,}"],
        ["Taille nœud", f"{layout['node_len']} octets"],
        ["Nœuds / secteur", str(layout["nodes_per_sector"])],
        ["Padding / secteur", f"{layout['wasted_space_per_sector_bytes']} octets"],
        ["", ""],
        ["Waste ratio moyen", f"{wr['mean']:.1%}"],
        ["Waste ratio médian", f"{wr['median']:.1%}"],
        ["Waste ratio P95", f"{wr['p95']:.1%}"],
        ["Read amp. moyen", f"{ra['mean']:.2f}×"],
        ["Read amp. médian", f"{ra['median']:.2f}×"],
        ["Localité spatiale", f"{summary['spatial_locality']['mean']:.1%}"],
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Métrique", "Valeur"],
        cellLoc="left",
        loc="center",
        colWidths=[0.5, 0.3],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)

    # Mise en forme
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#34495e")
            cell.set_text_props(color="white", fontweight="bold")
        elif rows[row - 1][0] == "":
            cell.set_facecolor("#ecf0f1")
        elif row >= 8:
            cell.set_facecolor("#fadbd8")

    fig.suptitle("Résumé — Gaspillage I/O DiskANN", fontsize=14, fontweight="bold")

    path = Path(output_dir) / "summary_table.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")


def main():
    parser = argparse.ArgumentParser(description="Visualise les résultats d'analyse I/O")
    parser.add_argument("--results", "-r", default="results",
                        help="Dossier contenant summary.json et per_query.json")
    parser.add_argument("--output", "-o", default=None,
                        help="Dossier de sortie pour les images (défaut: même que --results)")
    args = parser.parse_args()

    output_dir = args.output or args.results
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("Chargement des résultats ...")
    summary, per_query = load_results(args.results)
    print(f"  {summary['num_queries']} requêtes analysées\n")

    print("Génération des graphiques :")
    plot_waste_ratio_histogram(per_query, output_dir)
    plot_visited_vs_wasted(per_query, output_dir)
    plot_read_amplification_boxplot(per_query, output_dir)
    plot_sectors_vs_useful(per_query, output_dir)
    plot_summary_table(summary, output_dir)

    print(f"\nTerminé ✓ — {5} graphiques générés dans {output_dir}/")


if __name__ == "__main__":
    main()
