"""
visualize_module.py — Module de visualisation réutilisable
"""

import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


def plot_all(summary, per_query, output_dir):
    """Génère tous les graphiques en un appel."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    waste_ratios = [q["waste_ratio"] for q in per_query]
    visited = [q["nodes_visited"] for q in per_query]
    wasted = [q["wasted_nodes"] for q in per_query]
    ra_nodes = [q["read_amplification_nodes"] for q in per_query]
    localities = [q["spatial_locality"] for q in per_query]

    # 1. Histogramme waste ratio
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(waste_ratios, bins=50, color="#e74c3c", alpha=0.8, edgecolor="white")
    ax.axvline(np.mean(waste_ratios), color="black", linestyle="--", linewidth=2,
               label=f"Moyenne: {np.mean(waste_ratios):.1%}")
    ax.set_xlabel("Waste Ratio", fontsize=12)
    ax.set_ylabel("Nombre de requêtes", fontsize=12)
    ax.set_title("Distribution du gaspillage d'I/O par requête", fontsize=14)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "waste_ratio_histogram.png", dpi=150)
    plt.close(fig)

    # 2. Scatter visited vs wasted
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(visited, wasted, alpha=0.4, s=10, color="#3498db")
    if len(visited) > 1:
        z = np.polyfit(visited, wasted, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(visited), max(visited), 100)
        ax.plot(x_line, p(x_line), "r--", linewidth=2,
                label=f"Pente: {z[0]:.1f}")
    ax.set_xlabel("Nœuds visités", fontsize=12)
    ax.set_ylabel("Nœuds gaspillés", fontsize=12)
    ax.set_title("Corrélation visite ↔ gaspillage", fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "visited_vs_wasted.png", dpi=150)
    plt.close(fig)

    # 3. Boxplot read amplification
    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(ra_nodes, vert=True, patch_artist=True)
    bp["boxes"][0].set_facecolor("#e74c3c")
    ax.set_ylabel("Read Amplification (×)", fontsize=12)
    ax.set_title("Distribution de la Read Amplification", fontsize=14)
    ax.set_xticklabels(["Nœuds"])
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "read_amplification_boxplot.png", dpi=150)
    plt.close(fig)

    # 4. Localité spatiale
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(localities, bins=30, color="#2ecc71", alpha=0.8, edgecolor="white")
    ax.axvline(np.mean(localities), color="black", linestyle="--", linewidth=2,
               label=f"Moyenne: {np.mean(localities):.1%}")
    ax.set_xlabel("Localité spatiale", fontsize=12)
    ax.set_ylabel("Nombre de requêtes", fontsize=12)
    ax.set_title("Qualité de la co-localisation", fontsize=14)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "spatial_locality_histogram.png", dpi=150)
    plt.close(fig)

    print(f"  4 graphiques sauvegardés dans {output_dir}/")
