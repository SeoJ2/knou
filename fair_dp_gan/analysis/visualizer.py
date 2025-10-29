"""
Visualization Tools

Creates publication-quality visualizations for Fair-DP-GAN results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Optional
from mpl_toolkits.mplot3d import Axes3D
import os


class Visualizer:
    """
    Creates visualizations for Fair-DP-GAN experiments.

    Generates:
    1. Pareto frontier plots (3D)
    2. Training curves
    3. Fairness metrics comparison
    4. Sample quality visualization

    Args:
        output_dir: Directory to save plots (default: 'visualizations')
        style: Matplotlib style (default: 'seaborn-v0_8')
    """

    def __init__(
        self,
        output_dir: str = 'visualizations',
        style: str = 'seaborn-v0_8'
    ):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Set style
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')

        sns.set_palette("husl")

    def plot_pareto_frontier(
        self,
        results: List[Dict[str, float]],
        model_names: List[str],
        pareto_indices: Optional[List[int]] = None,
        save_path: Optional[str] = None
    ):
        """
        Plot 3D Pareto frontier for Privacy-Utility-Fairness tradeoff.

        Args:
            results: List of result dicts
            model_names: Names of models
            pareto_indices: Indices of Pareto-optimal solutions
            save_path: Path to save figure
        """
        # Extract objectives
        epsilons = [r.get('epsilon', 0) for r in results]
        utilities = [r.get('fid', r.get('feature_distance', 0)) for r in results]
        fairness = [r.get('spd', 0) for r in results]

        # Create 3D plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot all points
        colors = ['blue' if i not in (pareto_indices or []) else 'red'
                 for i in range(len(results))]
        sizes = [50 if i not in (pareto_indices or []) else 150
                for i in range(len(results))]

        scatter = ax.scatter(
            epsilons, utilities, fairness,
            c=colors, s=sizes, alpha=0.6, edgecolors='black'
        )

        # Add labels for Pareto-optimal points
        if pareto_indices:
            for idx in pareto_indices:
                ax.text(
                    epsilons[idx], utilities[idx], fairness[idx],
                    model_names[idx], fontsize=8
                )

        # Labels
        ax.set_xlabel('Privacy Loss (ε)', fontsize=12, labelpad=10)
        ax.set_ylabel('Utility Loss (FID/Distance)', fontsize=12, labelpad=10)
        ax.set_zlabel('Fairness Loss (SPD)', fontsize=12, labelpad=10)
        ax.set_title('Privacy-Utility-Fairness Pareto Frontier', fontsize=14, pad=20)

        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
                  markersize=8, label='Non-Pareto'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                  markersize=12, label='Pareto-Optimal')
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        plt.tight_layout()

        # Save
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'pareto_frontier.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved Pareto frontier plot to {save_path}")

    def plot_training_curves(
        self,
        histories: Dict[str, Dict[str, List]],
        save_path: Optional[str] = None
    ):
        """
        Plot training curves for multiple models.

        Args:
            histories: Dict of model_name -> training history
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        metrics = ['loss_g', 'loss_c', 'epsilon', 'fairness_loss']
        titles = ['Generator Loss', 'Critic Loss', 'Privacy Budget (ε)', 'Fairness Loss']

        for ax, metric, title in zip(axes.flat, metrics, titles):
            for model_name, history in histories.items():
                if metric in history:
                    values = history[metric]
                    ax.plot(values, label=model_name, linewidth=2)

            ax.set_xlabel('Epoch', fontsize=11)
            ax.set_ylabel(title, fontsize=11)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.output_dir, 'training_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved training curves to {save_path}")

    def plot_fairness_comparison(
        self,
        results: List[Dict[str, float]],
        model_names: List[str],
        save_path: Optional[str] = None
    ):
        """
        Create bar chart comparing fairness metrics across models.

        Args:
            results: List of result dicts
            model_names: Names of models
            save_path: Path to save figure
        """
        # Extract SPD values
        spd_values = [r.get('spd', 0) for r in results]

        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(model_names))
        bars = ax.bar(x, spd_values, color='steelblue', alpha=0.7, edgecolor='black')

        # Color Pareto-optimal bars differently
        # (Would need pareto_indices passed in, simplified here)

        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, spd_values)):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2, height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10
            )

        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Statistical Parity Difference (SPD)', fontsize=12)
        ax.set_title('Fairness Comparison Across Models', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.grid(True, axis='y', alpha=0.3)

        # Add horizontal line at SPD=0.1 (common fairness threshold)
        ax.axhline(y=0.1, color='red', linestyle='--', linewidth=2, label='Fairness Threshold (0.1)')
        ax.legend()

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.output_dir, 'fairness_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved fairness comparison to {save_path}")

    def plot_metrics_heatmap(
        self,
        results: List[Dict[str, float]],
        model_names: List[str],
        metrics: List[str] = None,
        save_path: Optional[str] = None
    ):
        """
        Create heatmap of all metrics across models.

        Args:
            results: List of result dicts
            model_names: Names of models
            metrics: List of metrics to include
            save_path: Path to save figure
        """
        if metrics is None:
            metrics = ['epsilon', 'fid', 'spd', 'puf_score']

        # Extract metric values
        data = []
        for result in results:
            row = []
            for metric in metrics:
                if metric == 'fid':
                    val = result.get('fid', result.get('feature_distance', 0))
                else:
                    val = result.get(metric, 0)
                row.append(val)
            data.append(row)

        data = np.array(data)

        # Normalize columns to [0, 1] for visualization
        data_norm = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0) + 1e-8)

        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(data_norm.T, cmap='RdYlGn_r', aspect='auto')

        # Set ticks
        ax.set_xticks(np.arange(len(model_names)))
        ax.set_yticks(np.arange(len(metrics)))
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_yticklabels(metrics)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Normalized Value', rotation=270, labelpad=15)

        # Add text annotations
        for i in range(len(metrics)):
            for j in range(len(model_names)):
                text = ax.text(
                    j, i, f'{data[j, i]:.2f}',
                    ha='center', va='center', color='black', fontsize=9
                )

        ax.set_title('Metrics Heatmap Across Models', fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.output_dir, 'metrics_heatmap.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved metrics heatmap to {save_path}")

    def plot_group_distribution(
        self,
        real_dist: List[float],
        fake_dist: List[float],
        group_names: List[str] = None,
        save_path: Optional[str] = None
    ):
        """
        Plot comparison of group distributions.

        Args:
            real_dist: Real data group distribution
            fake_dist: Generated data group distribution
            group_names: Names of groups
            save_path: Path to save figure
        """
        if group_names is None:
            group_names = [f'Group {i}' for i in range(len(real_dist))]

        x = np.arange(len(group_names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))

        bars1 = ax.bar(x - width/2, real_dist, width, label='Real Data',
                      color='steelblue', alpha=0.8, edgecolor='black')
        bars2 = ax.bar(x + width/2, fake_dist, width, label='Generated Data',
                      color='coral', alpha=0.8, edgecolor='black')

        ax.set_xlabel('Group', fontsize=12)
        ax.set_ylabel('Proportion', fontsize=12)
        ax.set_title('Group Distribution Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(group_names)
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.output_dir, 'group_distribution.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved group distribution plot to {save_path}")

    def __repr__(self) -> str:
        return f"Visualizer(output_dir={self.output_dir})"
