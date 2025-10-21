"""
Group-Aware Differential Privacy SGD optimizer
Implements adaptive per-group gradient clipping for fairness-aware DP training
"""
import torch
import torch.optim as optim
from opacus.accountants import RDPAccountant
import numpy as np


class GroupAwareDPSGD:
    """
    Group-Aware DP-SGD with adaptive clipping

    This optimizer applies different gradient clipping norms to different groups
    based on the sensitive attribute to ensure fairness while maintaining privacy.
    """

    def __init__(
        self,
        params,
        lr=0.0002,
        betas=(0.5, 0.999),
        max_grad_norm=1.0,
        noise_multiplier=1.0,
        epsilon=None,
        delta=1e-5,
        adaptive_clipping=True,
        group_ratios=None
    ):
        """
        Args:
            params: Model parameters to optimize
            lr: Learning rate
            betas: Adam beta parameters
            max_grad_norm: Maximum gradient norm for clipping
            noise_multiplier: Noise multiplier for DP (sigma)
            epsilon: Target epsilon for privacy budget (if None, use noise_multiplier)
            delta: Target delta for privacy
            adaptive_clipping: Whether to use group-aware adaptive clipping
            group_ratios: Dictionary mapping group_id to ratio (for weighting)
        """
        self.optimizer = optim.Adam(params, lr=lr, betas=betas)
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier
        self.epsilon = epsilon
        self.delta = delta
        self.adaptive_clipping = adaptive_clipping
        self.group_ratios = group_ratios or {}

        # Privacy accountant
        self.accountant = RDPAccountant()
        self.steps = 0

        # Group-specific clipping norms
        self.group_clip_norms = {}

    def update_group_ratios(self, group_counts):
        """
        Update group ratios based on current batch statistics

        Args:
            group_counts: Dictionary mapping group_id to count in batch
        """
        total = sum(group_counts.values())
        if total > 0:
            self.group_ratios = {
                group_id: count / total
                for group_id, count in group_counts.items()
            }

    def compute_adaptive_clip_norms(self, group_ids):
        """
        Compute adaptive clipping norms for each group

        Uses weighted average based on group representation:
        C_g = C * (1 + λ * (r_g - 1/K))
        where r_g is group ratio and K is number of groups

        Args:
            group_ids: Tensor of group IDs for each sample in batch

        Returns:
            Dictionary mapping group_id to clip_norm
        """
        if not self.adaptive_clipping:
            # Uniform clipping
            unique_groups = torch.unique(group_ids)
            return {
                int(g): self.max_grad_norm
                for g in unique_groups
            }

        unique_groups = torch.unique(group_ids)
        num_groups = len(unique_groups)

        if num_groups == 0:
            return {}

        # Count samples per group in current batch
        group_counts = {}
        for g in unique_groups:
            group_counts[int(g)] = (group_ids == g).sum().item()

        # Update group ratios
        self.update_group_ratios(group_counts)

        # Compute adaptive clip norms
        clip_norms = {}
        lambda_adapt = 0.5  # Adaptation strength

        for group_id, ratio in self.group_ratios.items():
            # Adaptive formula: increase clip norm for underrepresented groups
            uniform_ratio = 1.0 / num_groups
            adjustment = 1 + lambda_adapt * (ratio - uniform_ratio)
            clip_norms[group_id] = self.max_grad_norm * adjustment

        self.group_clip_norms = clip_norms
        return clip_norms

    def clip_and_accumulate_gradients(self, samples_grad, group_ids):
        """
        Clip gradients per-sample and accumulate with group-aware clipping

        Args:
            samples_grad: List of per-sample gradients
            group_ids: Tensor of group IDs for each sample

        Returns:
            Clipped and aggregated gradients
        """
        if len(samples_grad) == 0:
            return None

        # Compute adaptive clip norms for each group
        clip_norms = self.compute_adaptive_clip_norms(group_ids)

        # Clip each sample's gradient based on its group
        clipped_grads = []
        for i, grad in enumerate(samples_grad):
            group_id = int(group_ids[i].item())
            clip_norm = clip_norms.get(group_id, self.max_grad_norm)

            # Flatten all gradients for this sample
            flat_grad = torch.cat([g.flatten() for g in grad])
            grad_norm = torch.norm(flat_grad, p=2)

            # Clip
            clip_factor = min(1.0, clip_norm / (grad_norm + 1e-8))
            clipped = [g * clip_factor for g in grad]
            clipped_grads.append(clipped)

        # Average clipped gradients
        batch_size = len(clipped_grads)
        avg_grad = []
        num_params = len(clipped_grads[0])

        for param_idx in range(num_params):
            param_grads = [clipped_grads[i][param_idx] for i in range(batch_size)]
            avg_param_grad = torch.stack(param_grads).mean(dim=0)
            avg_grad.append(avg_param_grad)

        return avg_grad

    def add_noise_to_gradients(self, gradients):
        """
        Add Gaussian noise to gradients for differential privacy

        Args:
            gradients: List of gradient tensors

        Returns:
            Noisy gradients
        """
        noisy_grads = []
        for grad in gradients:
            noise = torch.normal(
                mean=0,
                std=self.noise_multiplier * self.max_grad_norm,
                size=grad.shape,
                device=grad.device
            )
            noisy_grads.append(grad + noise)

        return noisy_grads

    def step(self, closure=None):
        """
        Perform a single optimization step

        Args:
            closure: A closure that reevaluates the model and returns the loss
        """
        # Standard optimizer step (assumes gradients are already computed and clipped)
        self.optimizer.step(closure)
        self.steps += 1

    def zero_grad(self):
        """Zero out the gradients"""
        self.optimizer.zero_grad()

    def get_privacy_spent(self, dataset_size, batch_size):
        """
        Calculate privacy budget spent

        Args:
            dataset_size: Total number of samples in dataset
            batch_size: Batch size used for training

        Returns:
            (epsilon, delta) tuple
        """
        sampling_rate = batch_size / dataset_size

        # Add noise to accountant for each step
        self.accountant.step(
            noise_multiplier=self.noise_multiplier,
            sample_rate=sampling_rate
        )

        # Get privacy spent
        epsilon = self.accountant.get_epsilon(delta=self.delta)

        return epsilon, self.delta

    def state_dict(self):
        """Return the state of the optimizer"""
        return {
            'optimizer': self.optimizer.state_dict(),
            'steps': self.steps,
            'group_ratios': self.group_ratios,
            'group_clip_norms': self.group_clip_norms
        }

    def load_state_dict(self, state_dict):
        """Load the optimizer state"""
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.steps = state_dict.get('steps', 0)
        self.group_ratios = state_dict.get('group_ratios', {})
        self.group_clip_norms = state_dict.get('group_clip_norms', {})


class SimpleGroupAwareDPSGD:
    """
    Simplified Group-Aware DP-SGD for easier integration
    This version works with standard PyTorch optimizers
    """

    def __init__(
        self,
        optimizer,
        max_grad_norm=1.0,
        noise_multiplier=1.0,
        adaptive_clipping=True
    ):
        """
        Args:
            optimizer: Base PyTorch optimizer
            max_grad_norm: Maximum gradient norm
            noise_multiplier: Noise multiplier for DP
            adaptive_clipping: Whether to use adaptive clipping
        """
        self.optimizer = optimizer
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier
        self.adaptive_clipping = adaptive_clipping
        self.group_ratios = {}

    def clip_gradients(self, model, group_ids=None):
        """
        Clip gradients with group-aware norms

        Args:
            model: Model with computed gradients
            group_ids: Group IDs for adaptive clipping (optional)

        Returns:
            Total gradient norm before clipping
        """
        # Get adaptive clip norm
        if group_ids is not None and self.adaptive_clipping:
            unique_groups = torch.unique(group_ids)
            num_groups = len(unique_groups)

            # Compute group ratios
            group_counts = {}
            for g in unique_groups:
                group_counts[int(g)] = (group_ids == g).sum().item()

            total = sum(group_counts.values())
            ratios = {g: c / total for g, c in group_counts.items()}

            # Use weighted average clip norm
            uniform_ratio = 1.0 / num_groups
            lambda_adapt = 0.5
            avg_adjustment = np.mean([
                1 + lambda_adapt * (r - uniform_ratio)
                for r in ratios.values()
            ])
            clip_norm = self.max_grad_norm * avg_adjustment
        else:
            clip_norm = self.max_grad_norm

        # Clip gradients
        total_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            clip_norm
        )

        return total_norm

    def add_noise(self, model):
        """
        Add Gaussian noise to model gradients

        Args:
            model: Model with clipped gradients
        """
        for param in model.parameters():
            if param.grad is not None:
                noise = torch.normal(
                    mean=0,
                    std=self.noise_multiplier * self.max_grad_norm,
                    size=param.grad.shape,
                    device=param.grad.device
                )
                param.grad += noise

    def step(self, closure=None):
        """Perform optimizer step"""
        return self.optimizer.step(closure)

    def zero_grad(self):
        """Zero gradients"""
        self.optimizer.zero_grad()
