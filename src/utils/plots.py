"""Plotting helpers for benchmark results."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def accuracy_vs_window(df: pd.DataFrame, out_path: Path, dataset: str) -> None:
    sub = df[df["dataset"] == dataset]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for algo, g in sub.groupby("algo"):
        agg = g.groupby("window_s")["accuracy"].agg(["mean", "std"]).reset_index()
        ax.errorbar(agg["window_s"], agg["mean"], yerr=agg["std"], marker="o", label=algo)
    ax.set_xlabel("Window length (s)")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{dataset}: accuracy vs window length")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def itr_bar(df: pd.DataFrame, out_path: Path, dataset: str, window_s: float) -> None:
    sub = df[(df["dataset"] == dataset) & (np.isclose(df["window_s"], window_s))]
    if sub.empty:
        return
    agg = sub.groupby("algo")["itr_bpm"].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(agg["algo"], agg["mean"], yerr=agg["std"], capsize=4)
    ax.set_ylabel("ITR (bits/min)")
    ax.set_title(f"{dataset} ITR @ {window_s:g}s")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def confusion_matrix_plot(cm: np.ndarray, labels: list, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
