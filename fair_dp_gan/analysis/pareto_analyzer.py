"""
Pareto Frontier Analysis

Analyzes the multi-objective tradeoff between Privacy, Utility, and Fairness.
Identifies Pareto-optimal solutions and computes hypervolume metrics.
"""

import numpy as np
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull


class ParetoAnalyzer:
    """
    Analyzes Pareto frontiers for multi-objective optimization.

    In Fair-DP-GAN, we have three competing objectives:
    1. Privacy (higher ε = worse privacy)
    2. Utility (higher FID = worse utility)
    3. Fairness (higher SPD = worse fairness)

    Goal: Find solutions that are Pareto-optimal (no improvement in one
    objective without degrading another).

    Args:
        objectives: List of objective names (default: ['privacy', 'utility', 'fairness'])
        maximize: Which objectives to maximize (default: all False = minimize)
    """

    def __init__(
        self,
        objectives: List[str] = None,
        maximize: List[bool] = None
    ):
        if objectives is None:
            self.objectives = ['privacy', 'utility', 'fairness']
        else:
            self.objectives = objectives

        if maximize is None:
            self.maximize = [False] * len(self.objectives)
        else:
            self.maximize = maximize

    def find_pareto_frontier(
        self,
        results: List[Dict[str, float]],
        model_names: List[str]
    ) -> Tuple[List[int], List[Dict[str, float]]]:
        """
        Identify Pareto-optimal solutions.

        A solution is Pareto-optimal if no other solution is better
        in all objectives.

        Args:
            results: List of dicts containing objective values
            model_names: Names of models corresponding to results

        Returns:
            pareto_indices: Indices of Pareto-optimal solutions
            pareto_results: Pareto-optimal results
        """
        n_solutions = len(results)

        # Extract objective values
        objective_values = self._extract_objectives(results)

        # Find Pareto frontier
        is_pareto = np.ones(n_solutions, dtype=bool)

        for i in range(n_solutions):
            for j in range(n_solutions):
                if i == j:
                    continue

                # Check if j dominates i
                if self._dominates(objective_values[j], objective_values[i]):
                    is_pareto[i] = False
                    break

        # Get Pareto-optimal indices
        pareto_indices = np.where(is_pareto)[0].tolist()
        pareto_results = [results[i] for i in pareto_indices]

        return pareto_indices, pareto_results

    def _extract_objectives(
        self,
        results: List[Dict[str, float]]
    ) -> np.ndarray:
        """
        Extract objective values from results.

        Converts to minimization problem (negate if maximize=True).
        """
        n_solutions = len(results)
        n_objectives = len(self.objectives)

        values = np.zeros((n_solutions, n_objectives))

        for i, result in enumerate(results):
            for j, obj in enumerate(self.objectives):
                # Get value (with fallback)
                if obj == 'privacy':
                    val = result.get('epsilon', result.get('privacy_risk', 0))
                elif obj == 'utility':
                    val = result.get('fid', result.get('feature_distance', 0))
                elif obj == 'fairness':
                    val = result.get('spd', 0)
                else:
                    val = result.get(obj, 0)

                # Convert to minimization
                if self.maximize[j]:
                    val = -val

                values[i, j] = val

        return values

    def _dominates(self, a: np.ndarray, b: np.ndarray) -> bool:
        """
        Check if solution a dominates solution b.

        a dominates b if:
        - a is no worse than b in all objectives
        - a is strictly better than b in at least one objective
        """
        return np.all(a <= b) and np.any(a < b)

    def compute_hypervolume(
        self,
        results: List[Dict[str, float]],
        reference_point: np.ndarray = None
    ) -> float:
        """
        Compute hypervolume indicator.

        Hypervolume measures the volume of objective space dominated
        by the Pareto frontier. Higher is better.

        Args:
            results: List of results
            reference_point: Worst acceptable point (default: max values)

        Returns:
            Hypervolume value
        """
        objective_values = self._extract_objectives(results)

        if reference_point is None:
            # Use worst point as reference
            reference_point = np.max(objective_values, axis=0) * 1.1

        # Simple hypervolume calculation for 2D/3D
        if objective_values.shape[1] == 2:
            return self._hypervolume_2d(objective_values, reference_point)
        elif objective_values.shape[1] == 3:
            return self._hypervolume_3d(objective_values, reference_point)
        else:
            # For higher dimensions, use approximate method
            return 0.0

    def _hypervolume_2d(
        self,
        points: np.ndarray,
        reference: np.ndarray
    ) -> float:
        """Compute 2D hypervolume."""
        # Sort by first objective
        sorted_points = points[points[:, 0].argsort()]

        volume = 0.0
        prev_x = 0

        for point in sorted_points:
            if np.all(point < reference):
                width = reference[0] - point[0]
                height = reference[1] - point[1]
                volume += width * height

        return volume

    def _hypervolume_3d(
        self,
        points: np.ndarray,
        reference: np.ndarray
    ) -> float:
        """Approximate 3D hypervolume."""
        # Filter points dominated by reference
        valid_points = points[np.all(points < reference, axis=1)]

        if len(valid_points) == 0:
            return 0.0

        # Approximate using sum of boxes
        volume = 0.0

        for point in valid_points:
            box_volume = np.prod(reference - point)
            volume += box_volume

        # Normalize by number of points to avoid overcounting
        volume /= len(valid_points)

        return volume

    def compute_coverage(
        self,
        pareto_results: List[Dict[str, float]],
        all_results: List[Dict[str, float]]
    ) -> float:
        """
        Compute Pareto frontier coverage.

        Coverage = |Pareto set| / |All solutions|

        Args:
            pareto_results: Pareto-optimal results
            all_results: All results

        Returns:
            Coverage ratio
        """
        return len(pareto_results) / len(all_results)

    def rank_solutions(
        self,
        results: List[Dict[str, float]],
        weights: Dict[str, float] = None
    ) -> List[Tuple[int, float]]:
        """
        Rank solutions using weighted sum.

        Args:
            results: List of results
            weights: Weights for each objective (default: equal weights)

        Returns:
            List of (index, score) tuples, sorted by score
        """
        if weights is None:
            weights = {obj: 1.0 / len(self.objectives) for obj in self.objectives}

        scores = []

        for i, result in enumerate(results):
            score = 0.0

            for obj in self.objectives:
                if obj == 'privacy':
                    val = result.get('epsilon', 0)
                    # Lower epsilon is better
                    obj_score = 1 / (1 + val)
                elif obj == 'utility':
                    val = result.get('fid', result.get('feature_distance', 0))
                    # Lower FID/distance is better
                    obj_score = 1 / (1 + val)
                elif obj == 'fairness':
                    val = result.get('spd', 0)
                    # Lower SPD is better
                    obj_score = 1 - min(val, 1.0)
                else:
                    obj_score = result.get(obj, 0)

                score += weights.get(obj, 1.0) * obj_score

            scores.append((i, score))

        # Sort by score (descending)
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores

    def get_summary(
        self,
        results: List[Dict[str, float]],
        model_names: List[str]
    ) -> str:
        """
        Get formatted summary of Pareto analysis.

        Args:
            results: List of results
            model_names: Model names

        Returns:
            Formatted string
        """
        pareto_indices, pareto_results = self.find_pareto_frontier(
            results, model_names
        )

        summary = "=== Pareto Frontier Analysis ===\n\n"
        summary += f"Total Solutions: {len(results)}\n"
        summary += f"Pareto-Optimal Solutions: {len(pareto_indices)}\n"
        summary += f"Coverage: {len(pareto_indices) / len(results):.2%}\n\n"

        summary += "Pareto-Optimal Models:\n"
        for idx in pareto_indices:
            name = model_names[idx]
            result = results[idx]

            summary += f"\n  {name}:\n"
            summary += f"    Privacy (ε): {result.get('epsilon', 'N/A')}\n"
            summary += f"    Utility (FID): {result.get('fid', result.get('feature_distance', 'N/A'))}\n"
            summary += f"    Fairness (SPD): {result.get('spd', 'N/A')}\n"

        # Hypervolume
        hv = self.compute_hypervolume(results)
        summary += f"\nHypervolume: {hv:.4f}\n"

        # Best model by weighted sum
        rankings = self.rank_solutions(results)
        best_idx = rankings[0][0]
        best_score = rankings[0][1]

        summary += f"\nBest Overall (weighted): {model_names[best_idx]} (score: {best_score:.4f})\n"

        return summary
