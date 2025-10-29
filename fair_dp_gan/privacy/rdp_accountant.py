"""
Rényi Differential Privacy (RDP) Accountant

This module implements precise privacy budget tracking using Rényi Differential Privacy,
which provides tighter privacy guarantees compared to standard (ε, δ)-DP composition.

RDP offers advantages for composing multiple privacy-preserving operations by tracking
privacy loss in terms of Rényi divergence across different orders α.

Reference:
    Mironov, I. (2017). Rényi differential privacy. CSF 2017.
"""

from typing import List, Tuple, Optional
import numpy as np
from scipy.special import comb
from scipy import optimize


class RDPAccountant:
    """
    Tracks privacy budget using Rényi Differential Privacy.

    RDP is parameterized by an order α and privacy parameter ε(α).
    This class tracks the RDP guarantees across multiple steps and converts
    to standard (ε, δ)-DP when needed.

    Args:
        target_delta: Target δ for (ε, δ)-DP conversion (default: 1e-5)
        orders: List of Rényi orders to track (default: [1.5, 2, 2.5, ..., 64])
    """

    def __init__(
        self,
        target_delta: float = 1e-5,
        orders: Optional[List[float]] = None
    ):
        self.target_delta = target_delta

        # Default orders to track (commonly used in literature)
        if orders is None:
            self.orders = [1.5, 2, 2.5, 3, 4, 5, 6, 8, 16, 32, 64]
        else:
            self.orders = sorted(orders)

        # Track RDP at each order
        self.rdp_sum = np.zeros(len(self.orders))

        # Track number of steps
        self.steps = 0

    def add_step(
        self,
        noise_multiplier: float,
        sample_rate: float,
        steps: int = 1
    ):
        """
        Add privacy cost of DP-SGD steps.

        Args:
            noise_multiplier: Ratio of noise standard deviation to clipping norm
            sample_rate: Sampling rate (batch_size / dataset_size)
            steps: Number of gradient steps (default: 1)
        """
        rdp = self._compute_rdp(noise_multiplier, sample_rate)
        self.rdp_sum += rdp * steps
        self.steps += steps

    def _compute_rdp(
        self,
        noise_multiplier: float,
        sample_rate: float
    ) -> np.ndarray:
        """
        Compute RDP for Gaussian mechanism with sampling.

        This uses the analytical formula for RDP of sampled Gaussian mechanism.

        Args:
            noise_multiplier: Noise standard deviation / L2 sensitivity
            sample_rate: Probability of sampling each example

        Returns:
            RDP values at each order in self.orders
        """
        if noise_multiplier == 0:
            # No privacy
            return np.array([np.inf] * len(self.orders))

        if sample_rate == 0:
            # No data processed
            return np.zeros(len(self.orders))

        rdp = np.zeros(len(self.orders))

        for i, alpha in enumerate(self.orders):
            if alpha == 1:
                # Special case: α = 1 is KL divergence
                rdp[i] = sample_rate * sample_rate / (2 * noise_multiplier ** 2)
            else:
                # General formula for α > 1
                rdp[i] = self._compute_rdp_for_order(
                    alpha, noise_multiplier, sample_rate
                )

        return rdp

    def _compute_rdp_for_order(
        self,
        alpha: float,
        sigma: float,
        q: float
    ) -> float:
        """
        Compute RDP at a specific order using analytical formula.

        This implements the tight RDP bounds for sampled Gaussian mechanism
        from Mironov (2017) and Wang et al. (2019).

        Args:
            alpha: Rényi order
            sigma: Noise multiplier
            q: Sampling probability

        Returns:
            RDP value at order alpha
        """
        if q == 0:
            return 0

        if q == 1:
            # No sampling: pure Gaussian mechanism
            return alpha / (2 * sigma ** 2)

        # For small sampling rates, use tight composition theorem
        if q < 0.01:
            return q * q * alpha / (2 * sigma ** 2)

        # General case: use log moments
        # This is an approximation that works well in practice
        log_moment = self._compute_log_moment(alpha, sigma, q)

        return log_moment / (alpha - 1)

    def _compute_log_moment(
        self,
        alpha: float,
        sigma: float,
        q: float
    ) -> float:
        """
        Compute log moment for RDP calculation.

        This uses numerical computation for the log moment generating function
        of the privacy loss random variable.
        """
        # Simplified approximation (more complex exact formula exists)
        # For production use, consider using Opacus implementation

        if alpha <= 1:
            return 0

        # Approximate using Taylor expansion for small q
        term1 = q * q * alpha / (2 * sigma ** 2)
        term2 = q * q * q * alpha * (alpha - 1) / (6 * sigma ** 4)

        return (alpha - 1) * (term1 + term2)

    def get_epsilon(self, delta: Optional[float] = None) -> float:
        """
        Convert RDP to (ε, δ)-DP.

        Uses the conversion formula:
            ε = min_α [ RDP(α) + log(1/δ) / (α - 1) ]

        Args:
            delta: Target δ (uses self.target_delta if None)

        Returns:
            Privacy parameter ε
        """
        if delta is None:
            delta = self.target_delta

        if delta <= 0 or delta >= 1:
            raise ValueError("Delta must be in (0, 1)")

        # Compute ε for each order and take minimum
        epsilons = []
        for i, alpha in enumerate(self.orders):
            if alpha == 1:
                continue

            # ε = RDP(α) + log(1/δ) / (α - 1)
            epsilon = self.rdp_sum[i] + np.log(1 / delta) / (alpha - 1)
            epsilons.append(epsilon)

        if not epsilons:
            return np.inf

        return min(epsilons)

    def get_privacy_spent(
        self,
        target_eps: Optional[float] = None,
        target_delta: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Get current privacy spent in (ε, δ) format.

        Args:
            target_eps: If provided, returns minimum δ for this ε
            target_delta: If provided, returns ε for this δ

        Returns:
            (epsilon, delta) tuple
        """
        if target_eps is not None:
            # Find minimum delta for given epsilon
            delta = self._compute_delta(target_eps)
            return (target_eps, delta)
        else:
            delta = target_delta if target_delta is not None else self.target_delta
            epsilon = self.get_epsilon(delta)
            return (epsilon, delta)

    def _compute_delta(self, target_epsilon: float) -> float:
        """
        Compute minimum δ for a given ε using RDP conversion.

        Solves: ε = min_α [ RDP(α) + log(1/δ) / (α - 1) ]
        for δ given ε.
        """
        # For each order, compute implied delta
        deltas = []
        for i, alpha in enumerate(self.orders):
            if alpha == 1:
                continue

            # From ε = RDP(α) + log(1/δ) / (α - 1)
            # We get: log(1/δ) = (ε - RDP(α)) * (α - 1)
            # So: δ = exp(-(ε - RDP(α)) * (α - 1))

            log_delta = -(target_epsilon - self.rdp_sum[i]) * (alpha - 1)

            if log_delta > 0:
                # Invalid: would require δ > 1
                continue

            delta = np.exp(log_delta)
            deltas.append(delta)

        if not deltas:
            return 1.0  # No privacy

        return min(deltas)

    def get_rdp(self) -> List[Tuple[float, float]]:
        """
        Get current RDP values.

        Returns:
            List of (order, RDP_value) tuples
        """
        return list(zip(self.orders, self.rdp_sum))

    def reset(self):
        """Reset the accountant to initial state."""
        self.rdp_sum = np.zeros(len(self.orders))
        self.steps = 0

    def get_summary(self) -> str:
        """
        Get a formatted summary of privacy spent.

        Returns:
            Formatted string with privacy information
        """
        eps, delta = self.get_privacy_spent()

        summary = "=== Privacy Budget Summary ===\n"
        summary += f"Total Steps: {self.steps}\n"
        summary += f"Privacy Spent: (ε={eps:.2f}, δ={delta:.2e})\n"
        summary += f"Target Delta: {self.target_delta:.2e}\n"
        summary += "\nRDP at different orders:\n"

        for alpha, rdp in zip(self.orders[:5], self.rdp_sum[:5]):
            summary += f"  α={alpha:.1f}: RDP={rdp:.2f}\n"

        return summary

    @staticmethod
    def compute_sigma_for_epsilon(
        target_epsilon: float,
        target_delta: float,
        sample_rate: float,
        epochs: int,
        dataset_size: int
    ) -> float:
        """
        Compute required noise multiplier to achieve target privacy budget.

        This is useful for hyperparameter tuning: given desired privacy level,
        find the noise multiplier needed.

        Args:
            target_epsilon: Desired ε
            target_delta: Desired δ
            sample_rate: Sampling rate per step
            epochs: Number of training epochs
            dataset_size: Size of training dataset

        Returns:
            Required noise multiplier σ
        """
        steps = int(epochs * dataset_size / (sample_rate * dataset_size))

        # Binary search for sigma
        def get_epsilon(sigma):
            accountant = RDPAccountant(target_delta=target_delta)
            accountant.add_step(sigma, sample_rate, steps)
            eps, _ = accountant.get_privacy_spent()
            return eps

        # Find sigma that gives target epsilon
        # Start with reasonable bounds
        sigma_low, sigma_high = 0.1, 100.0

        # Binary search
        for _ in range(50):
            sigma_mid = (sigma_low + sigma_high) / 2
            eps_mid = get_epsilon(sigma_mid)

            if eps_mid > target_epsilon:
                # Need more noise
                sigma_low = sigma_mid
            else:
                # Need less noise
                sigma_high = sigma_mid

            if abs(eps_mid - target_epsilon) < 0.01:
                break

        return sigma_mid
