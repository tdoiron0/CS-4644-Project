#!/usr/bin/env python3
"""
Plot training and validation metrics from a metrics.csv file.
Usage: python scripts/plot_metrics.py outputs/run_<timestamp>/metrics.csv
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_train_val_loss(df, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["epoch"], df["train_loss"], "o-", label="Train Loss")
    ax.plot(df["epoch"], df["val_loss"], "s-", label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Train vs Val Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "loss.png"), dpi=150)
    plt.close(fig)


def plot_train_val_perplexity(df, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["epoch"], df["train_ppl"], "o-", label="Train PPL")
    ax.plot(df["epoch"], df["val_ppl"], "s-", label="Val PPL")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Perplexity")
    ax.set_title("Train vs Val Perplexity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "perplexity.png"), dpi=150)
    plt.close(fig)


def plot_train_accuracy(df, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["epoch"], df["train_acc_overall"], "o-", label="Overall", linewidth=2)
    ax.plot(df["epoch"], df["train_acc_manufacturer"], "^--", label="Manufacturer")
    ax.plot(df["epoch"], df["train_acc_family"], "v--", label="Family")
    ax.plot(df["epoch"], df["train_acc_variant"], "D--", label="Variant")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Train Accuracy per Category")
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "train_accuracy.png"), dpi=150)
    plt.close(fig)


def plot_val_accuracy(df, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["epoch"], df["val_acc_overall"], "o-", label="Overall", linewidth=2)
    ax.plot(df["epoch"], df["val_acc_manufacturer"], "^--", label="Manufacturer")
    ax.plot(df["epoch"], df["val_acc_family"], "v--", label="Family")
    ax.plot(df["epoch"], df["val_acc_variant"], "D--", label="Variant")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (Exact Match)")
    ax.set_title("Validation Accuracy per Category")
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "val_accuracy.png"), dpi=150)
    plt.close(fig)


def plot_val_loss_per_category(df, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["epoch"], df["val_loss"], "o-", label="Overall", linewidth=2)
    ax.plot(df["epoch"], df["val_loss_manufacturer"], "^--", label="Manufacturer")
    ax.plot(df["epoch"], df["val_loss_family"], "v--", label="Family")
    ax.plot(df["epoch"], df["val_loss_variant"], "D--", label="Variant")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Validation Loss per Category")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "val_loss_per_category.png"), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot training metrics")
    parser.add_argument("csv_path", help="Path to metrics.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    out_dir = os.path.dirname(args.csv_path)

    plot_train_val_loss(df, out_dir)
    plot_train_val_perplexity(df, out_dir)
    plot_train_accuracy(df, out_dir)
    plot_val_accuracy(df, out_dir)
    plot_val_loss_per_category(df, out_dir)

    print(f"Plots saved to {out_dir}:")
    print("  loss.png")
    print("  perplexity.png")
    print("  train_accuracy.png")
    print("  val_accuracy.png")
    print("  val_loss_per_category.png")


if __name__ == "__main__":
    main()
