"""
Gradient-based saliency visualization for PPO agents.

Visualizes what regions of the input observation contribute most to:
- Value function estimation (∂V(s)/∂input)
- Policy decisions (∂logit[action]/∂input)

Uses threshold-based masking to show only the regions the agent attends to.
Inspired by the visualization approach in the Dueling Networks paper.
"""

import numpy as np
import torch
from typing import Optional, Tuple, List, Dict
import wandb


def compute_value_gradient(agent, obs: torch.Tensor) -> torch.Tensor:
    """
    Compute gradient of value function w.r.t. input observation.
    
    Args:
        agent: PPO agent with get_value method
        obs: Input observation tensor (B, C, H, W), requires_grad will be set
        
    Returns:
        Gradient tensor of same shape as obs (B, C, H, W)
    """
    obs = obs.clone().detach().requires_grad_(True)

    # Forward pass through value function
    value = agent.get_value(obs)

    # Sum values across batch for backward (each gets its own gradient)
    value_sum = value.sum()

    # Backward pass
    value_sum.backward()

    return obs.grad


def compute_policy_gradient(agent, obs: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute gradient of policy logit w.r.t. input observation.

    If action is provided, computes gradient for that action's logit.
    Otherwise, computes gradient for the argmax action (greedy policy).

    Args:
        agent: PPO agent with get_action_and_value method
        obs: Input observation tensor (B, C, H, W), requires_grad will be set
        action: Optional action tensor (B,) - if None, uses argmax

    Returns:
        Gradient tensor of same shape as obs (B, C, H, W)
    """
    obs = obs.clone().detach().requires_grad_(True)

    # Forward pass to get logits
    _, _, _, _, logits = agent.get_action_and_value(obs)

    # If no action provided, use the greedy action
    if action is None:
        action = logits.argmax(dim=-1)

    # Get logits for the selected actions
    batch_size = obs.shape[0]
    selected_logits = logits[torch.arange(batch_size, device=obs.device), action]

    # Sum for backward
    selected_logits_sum = selected_logits.sum()

    # Backward pass
    selected_logits_sum.backward()

    return obs.grad


def normalize_gradient(gradient: torch.Tensor, percentile: float = 99) -> torch.Tensor:
    """
    Normalize gradient to [0, 1] range with percentile clipping for robustness.
    Keeps the same shape as input (C, H, W) or (B, C, H, W).

    Args:
        gradient: Gradient tensor (C, H, W) or (B, C, H, W)
        percentile: Clip absolute values above this percentile

    Returns:
        Normalized gradient in [0, 1] with same shape
    """
    # Take absolute value (we care about magnitude, not sign)
    grad_abs = gradient.abs()

    # Flatten for percentile computation
    flat = grad_abs.flatten()

    # Compute percentile threshold
    if percentile < 100:
        threshold = torch.quantile(flat, percentile / 100.0)
        grad_abs = torch.clamp(grad_abs, max=threshold)

    # Min-max normalization
    min_val = grad_abs.min()
    max_val = grad_abs.max()

    if max_val - min_val > 1e-8:
        normalized = (grad_abs - min_val) / (max_val - min_val)
    else:
        normalized = torch.zeros_like(grad_abs)

    return normalized


def gradient_to_image(gradient: torch.Tensor) -> np.ndarray:
    """
    Convert normalized gradient tensor to displayable RGB image.

    Args:
        gradient: Normalized gradient tensor (C, H, W) in [0, 1]

    Returns:
        Image array (H, W, 3) in [0, 255] uint8
    """
    if gradient.dim() == 4:
        gradient = gradient[0]  # Take first in batch

    # (C, H, W) -> (H, W, C)
    image = gradient.cpu().numpy().transpose(1, 2, 0)

    # Handle grayscale (stack to RGB)
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)

    # Scale to [0, 255]
    image = (image * 255).astype(np.uint8)

    return image


def create_masked_composite(image: np.ndarray, gradient: np.ndarray, threshold: float) -> np.ndarray:
    """
    Create a 3-panel composite:
    1. Heatmap
    2. Attended Areas (Low attention masked with Pink)
    3. Ignored Areas (High attention masked with Green)
    """
    import matplotlib.pyplot as plt

    # Ensure gradient is 2D (H, W)
    if gradient.ndim == 3:
        # average across channels if needed, assuming (C, H, W) from tensor or (H, W, C) from something else?
        # In create_images, value_grad_np is from value_grad.numpy(). value_grad is (C, H, W).
        # So gradient here is (C, H, W).
        grad_mag = np.mean(gradient, axis=0) # (H, W)
    else:
        grad_mag = gradient

    # Normalize 0-1 for heatmap
    grad_norm = (grad_mag - grad_mag.min()) / (grad_mag.max() - grad_mag.min() + 1e-8)

    # 1. Heatmap
    cmap = plt.get_cmap('jet')
    heatmap = (cmap(grad_norm)[:, :, :3] * 255).astype(np.uint8)

    # 2. Attended Areas (Mask out low attention)
    # Binary mask: 1 if attended (high grad), 0 otherwise
    # Threshold is passed from Visualizer, usually relative or absolute.
    # If threshold is small (e.g. 0.01), it assumes gradient is somewhat scaled.
    # The gradient passed to create_images is usually normalized by normalize_gradient.

    mask = (grad_mag > threshold).astype(bool)

    # Pink color for masked out (unattended) areas
    pink = np.array([255, 20, 147], dtype=np.uint8) # DeepPink#
    lightpink = np.array([255, 182, 193], dtype=np.uint8) # LightPink
    white = np.array([255, 255, 255], dtype=np.uint8) # White
    lightgray = np.array([128, 128, 128], dtype=np.uint8)
    # an even lighter gray for the ignored areas, to make the contrast clearer
    lightgray = np.array([200, 200, 200], dtype=np.uint8)

    # use a nice blue for attended areas instead of leaving them unchanged, to make the contrast clearer
    blue = np.array([0, 0, 255], dtype=np.uint8)
    orange = np.array([255, 165, 0], dtype=np.uint8)

    attended_img = image.copy()
    # Where mask is False (unattended), set to pink
    attended_img[~mask] = lightgray

    # 3. Ignored Areas (Mask out high attention)
    # Green color for masked out (attended) areas
    green = np.array([0, 255, 0], dtype=np.uint8) # Lime Green

    ignored_img = image.copy()
    # Where mask is True (attended), set to green
    ignored_img[mask] = green

    # Combine
    sep = np.ones((image.shape[0], 2, 3), dtype=np.uint8) * 255
    # combined = np.concatenate([image, sep, attended_img, sep, ignored_img, sep, heatmap], axis=1)
    combined = np.concatenate([image, sep, attended_img, sep, heatmap], axis=1)

    return combined


def create_masked_image(
    image: np.ndarray,
    gradient: np.ndarray,
    threshold: float = 0.01,
    soft_mask: bool = False,
    background_brightness: float = 0.0
) -> tuple:
    """
    Mask the image to show only high-gradient regions (what the agent attends to).

    Args:
        image: Original image (H, W, 3) in [0, 255] uint8
        gradient: Normalized gradient (C, H, W) or (H, W, C) or (H, W) in [0, 1]
        threshold: Direct threshold value in [0, 1] for the normalized gradient.
                   Pixels with gradient > threshold are shown.
                   e.g., 0.01 = show pixels where gradient > 1% of max
        soft_mask: If True, use smooth mask. If False, use hard binary mask.
        background_brightness: How bright the non-attended regions are (0=black, 1=full)

    Returns:
        Tuple of (masked_image, mask):
            - masked_image: (H, W, 3) in [0, 255] uint8
            - mask: (H, W) in [0, 1] float - the binary/soft mask used
    """
    # Convert gradient to (H, W) by taking max across channels (not mean!)
    # This way if ANY channel has high gradient, that pixel is attended
    if gradient.ndim == 3:
        # Check if CHW or HWC format
        if gradient.shape[0] in [1, 3, 4]:  # CHW format
            grad_magnitude = np.abs(gradient).max(axis=0)  # (H, W)
        else:  # HWC format
            grad_magnitude = np.abs(gradient).max(axis=-1)  # (H, W)
    else:
        grad_magnitude = np.abs(gradient)

    if soft_mask:
        # Soft mask: smoothly transition from background to foreground
        # Values below threshold get background_brightness, above get scaled up to 1
        max_val = grad_magnitude.max()
        if max_val > threshold:
            mask = np.clip((grad_magnitude - threshold) / (max_val - threshold + 1e-8), 0, 1)
        else:
            mask = np.zeros_like(grad_magnitude)
        mask = background_brightness + mask * (1 - background_brightness)
    else:
        # Hard binary mask: gradient > threshold -> show, else -> background
        mask = np.where(grad_magnitude > threshold, 1.0, background_brightness)

    # Expand mask to 3 channels
    mask_3ch = mask[:, :, np.newaxis]

    # Apply mask to image
    masked = image.astype(np.float32) * mask_3ch

    return np.clip(masked, 0, 255).astype(np.uint8), mask


def create_attention_visualization(
    image: np.ndarray,
    gradient: np.ndarray,
    threshold: float = 0.01,
    show_heatmap: bool = True,
    colormap: str = 'hot'
) -> np.ndarray:
    """
    Create visualization showing: original | masked attention | heatmap

    Args:
        image: Original image (H, W, 3)
        gradient: Normalized gradient (C, H, W) or (H, W)
        threshold: Direct threshold in [0, 1] for masking
        show_heatmap: Whether to include heatmap panel
        colormap: Colormap for heatmap

    Returns:
        Concatenated visualization
    """
    import matplotlib.pyplot as plt

    separator = np.ones((image.shape[0], 2, 3), dtype=np.uint8) * 255

    # Create masked image (what agent attends to)
    masked = create_masked_image(
        image, gradient,
        threshold=threshold,
        soft_mask=False,
        background_brightness=0.0
    )

    panels = [image, separator, masked]

    if show_heatmap:
        # Create heatmap
        if gradient.ndim == 3:
            if gradient.shape[0] in [1, 3, 4]:  # CHW
                grad_magnitude = np.abs(gradient).mean(axis=0)
            else:  # HWC
                grad_magnitude = np.abs(gradient).mean(axis=-1)
        else:
            grad_magnitude = np.abs(gradient)

        cmap = plt.get_cmap(colormap)
        heatmap = cmap(grad_magnitude)[:, :, :3]
        heatmap = (heatmap * 255).astype(np.uint8)

        panels.extend([separator, heatmap])

    return np.concatenate(panels, axis=1)


def create_threshold_comparison(
    image: np.ndarray,
    gradient: np.ndarray,
    thresholds: List[float] = [0.005, 0.01, 0.05, 0.1]
) -> np.ndarray:
    """
    Create side-by-side comparison of different threshold levels.

    Args:
        image: Original image (H, W, 3)
        gradient: Normalized gradient (C, H, W) or (H, W)
        thresholds: List of threshold values in [0, 1] to compare

    Returns:
        Concatenated image showing original + each threshold level
    """
    separator = np.ones((image.shape[0], 2, 3), dtype=np.uint8) * 255

    panels = [image, separator]
    for thresh in thresholds:
        masked = create_masked_image(
            image, gradient,
            threshold=thresh,
            soft_mask=False,
            background_brightness=0.0
        )
        panels.extend([masked, separator])

    return np.concatenate(panels[:-1], axis=1)  # Remove last separator


def create_combined_attention_view(
    image: np.ndarray,
    value_grad: np.ndarray,
    policy_grad: np.ndarray,
    threshold: float = 0.01
) -> np.ndarray:
    """
    Create combined view: original | value_attention | policy_attention | value_heatmap | policy_heatmap

    Shows what the value function and policy are attending to in the image.

    Args:
        image: Original image (H, W, 3)
        value_grad: Value gradient (C, H, W) or (H, W)
        policy_grad: Policy gradient (C, H, W) or (H, W)
        threshold: Direct threshold in [0, 1] for masking

    Returns:
        Concatenated image (H, W*5, 3)
    """
    import matplotlib.pyplot as plt

    separator = np.ones((image.shape[0], 2, 3), dtype=np.uint8) * 255

    # Create masked images showing attention
    value_masked = create_masked_image(
        image, value_grad,
        threshold=threshold,
        soft_mask=False,
        background_brightness=0.0
    )

    policy_masked = create_masked_image(
        image, policy_grad,
        threshold=threshold,
        soft_mask=False,
        background_brightness=0.0
    )

    # Create heatmaps
    def to_heatmap(grad):
        if grad.ndim == 3:
            if grad.shape[0] in [1, 3, 4]:
                mag = np.abs(grad).mean(axis=0)
            else:
                mag = np.abs(grad).mean(axis=-1)
        else:
            mag = np.abs(grad)
        cmap = plt.get_cmap('hot')
        hm = cmap(mag)[:, :, :3]
        return (hm * 255).astype(np.uint8)

    value_heatmap = to_heatmap(value_grad)
    policy_heatmap = to_heatmap(policy_grad)

    return np.concatenate([
        image, separator,
        value_masked, separator,
        policy_masked, separator,
        value_heatmap, separator,
        policy_heatmap
    ], axis=1)


def obs_to_image(obs: torch.Tensor) -> np.ndarray:
    """
    Convert observation tensor to displayable image.

    Args:
        obs: Observation tensor (C, H, W) in [0, 255]

    Returns:
        Image array (H, W, 3) in [0, 255] uint8
    """
    if obs.dim() == 4:
        obs = obs[0]  # Take first in batch

    # (C, H, W) -> (H, W, C)
    image = obs.cpu().numpy().transpose(1, 2, 0)

    # Handle grayscale (stack to RGB)
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)

    return np.clip(image, 0, 255).astype(np.uint8)


def add_label_to_image(
    image: np.ndarray,
    label: str,
    font_scale: float = 0.4,
    thickness: int = 1,
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    text_color: Tuple[int, int, int] = (255, 255, 255),
    padding: int = 4
) -> np.ndarray:
    """
    Add a text label bar at the top of an image.

    Args:
        image: Input image (H, W, 3)
        label: Text to display
        font_scale: Font size scale
        thickness: Font thickness
        bg_color: Background color of label bar (B, G, R)
        text_color: Text color (B, G, R)
        padding: Padding around text

    Returns:
        Image with label bar at top (H + label_height, W, 3)
    """
    try:
        import cv2
        has_cv2 = True
    except ImportError:
        has_cv2 = False

    img_h, img_w = image.shape[:2]

    if has_cv2:
        # Use OpenCV for text rendering (higher quality)
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        label_height = text_h + baseline + 2 * padding

        # Create label bar
        label_bar = np.full((label_height, img_w, 3), bg_color, dtype=np.uint8)

        # Center text
        text_x = (img_w - text_w) // 2
        text_y = padding + text_h

        cv2.putText(label_bar, label, (text_x, text_y), font, font_scale, text_color, thickness)
    else:
        # Fallback: simple label bar without text (just colored bar)
        # This is a basic fallback if cv2 is not available
        label_height = 16
        label_bar = np.full((label_height, img_w, 3), bg_color, dtype=np.uint8)

        # Draw a simple indicator line at different positions for different labels
        # This is just a visual fallback
        label_bar[label_height//2 - 1:label_height//2 + 1, :, :] = text_color

    # Stack label on top of image
    return np.vstack([label_bar, image])


class GradientVisualizer:
    """
    Collects frames and gradients during rollout for video creation.
    Tracks both value and policy gradients.
    Uses threshold-based masking to show what the agent attends to.
    """

    def __init__(
        self,
        agent,
        device: torch.device,
        n_episodes: int = 3,
        max_frames_per_episode: int = 500,
        threshold: float = 0.01,
        background_brightness: float = 0.0,
    ):
        """
        Args:
            agent: PPO agent
            device: Torch device
            n_episodes: Number of episodes to visualize
            max_frames_per_episode: Max frames to store per episode
            threshold: Direct threshold in [0, 1] for attention masking.
                      Pixels with normalized gradient > threshold are shown.
                      e.g., 0.01 means show where gradient > 1% of max
            background_brightness: Brightness of non-attended regions (0=black, 1=full)
        """
        self.agent = agent
        self.device = device
        self.n_episodes = n_episodes
        self.max_frames_per_episode = max_frames_per_episode
        self.threshold = threshold
        self.background_brightness = background_brightness

        # Storage for collected frames
        self.episodes_data: List[List[Dict]] = []
        self.current_episode_frames: List[Dict] = []
        self.episodes_collected = 0

    def reset(self):
        """Reset for new evaluation run."""
        self.episodes_data = []
        self.current_episode_frames = []
        self.episodes_collected = 0

    def is_collecting(self) -> bool:
        """Check if we should still collect episodes."""
        return self.episodes_collected < self.n_episodes

    def step(self, obs: torch.Tensor, done: bool, env_idx: int = 0, action: Optional[torch.Tensor] = None):
        """
        Process a single step during rollout.

        Args:
            obs: Current observation (C, H, W) or (1, C, H, W)
            done: Whether episode ended
            env_idx: Environment index (for parallel envs, we track env 0)
            action: Optional action taken (for policy gradient)
        """
        if not self.is_collecting():
            return

        if env_idx != 0:
            return

        # Handle episode end FIRST (before frame limit check)
        if done:
            n_frames = len(self.current_episode_frames)
            if n_frames > 10:  # Only keep episodes with >10 frames
                self.episodes_data.append(list(self.current_episode_frames))
                self.episodes_collected += 1
            self.current_episode_frames = []
            return

        # Only collect frames if we haven't hit frame limit
        if len(self.current_episode_frames) >= self.max_frames_per_episode:
            return

        # Ensure correct shape
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)

        self.agent.eval()

        # Compute value gradient
        value_gradient = compute_value_gradient(self.agent, obs)
        value_gradient_normalized = normalize_gradient(value_gradient[0])

        # Compute policy gradient
        policy_gradient = compute_policy_gradient(self.agent, obs, action)
        policy_gradient_normalized = normalize_gradient(policy_gradient[0])

        # Store frame data
        self.current_episode_frames.append({
            'obs': obs[0].detach().cpu(),
            'value_gradient': value_gradient_normalized.detach().cpu(),
            'policy_gradient': policy_gradient_normalized.detach().cpu(),
        })

    def create_images(self, max_frames=20):
        """
        Create visualization images and compute sparsity stats.

        Args:
            max_frames: Max number of frames to generate visualization images for.

        Returns:
            Dictionary with:
            - 'value': List of images
            - 'policy': List of images
            - 'combined': List of images
            - 'value_mask': List of images
            - 'policy_mask': List of images
            - 'value_mask_sparsity': float
            - 'policy_mask_sparsity': float
        """
        import matplotlib.pyplot as plt
        from typing import Any

        if not self.episodes_data:
            return {
                'value': [], 'policy': [], 'combined': [],
                'value_mask': [], 'policy_mask': [],
                'value_mask_sparsity': 0.0, 'policy_mask_sparsity': 0.0,
            }

        value_images = []
        policy_images = []
        combined_images = []
        value_mask_images = []
        policy_mask_images = []
        value_attention_images = []
        policy_attention_images = []

        # Track mask statistics
        value_mask_zeros = []
        policy_mask_zeros = []
        total_pixels = None

        frames_generated = 0

        for episode in self.episodes_data:
            for frame_data in episode:
                obs = frame_data['obs']
                value_grad = frame_data['value_gradient']
                policy_grad = frame_data['policy_gradient']

                # Convert to images
                image = obs_to_image(obs)
                img_h, img_w = image.shape[:2]

                if total_pixels is None:
                    total_pixels = img_h * img_w

                # Convert gradients to numpy (C, H, W)
                value_grad_np = value_grad.numpy()
                policy_grad_np = policy_grad.numpy()

                # Create masked attention images (returns mask too)
                _, value_mask = create_masked_image(
                    image, value_grad_np,
                    threshold=self.threshold,
                    soft_mask=False,
                    background_brightness=self.background_brightness
                )

                _, policy_mask = create_masked_image(
                    image, policy_grad_np,
                    threshold=self.threshold,
                    soft_mask=False,
                    background_brightness=self.background_brightness
                )

                # Count zero pixels in masks (where mask <= background_brightness)
                value_zeros = np.sum(value_mask <= self.background_brightness)
                policy_zeros = np.sum(policy_mask <= self.background_brightness)
                value_mask_zeros.append(value_zeros)
                policy_mask_zeros.append(policy_zeros)

                if frames_generated < max_frames:
                    # Create the new composite visualizations
                    value_composite = create_masked_composite(image, value_grad_np, self.threshold)
                    policy_composite = create_masked_composite(image, policy_grad_np, self.threshold)

                    value_images.append(value_composite)
                    policy_images.append(policy_composite)

                    # Store masks separately (raw) for old compat if needed, or derived from mask
                    # Keeping original logic for masks roughly, but we don't need combined_images as requested logic
                    # implies value/policy composites are the main thing now.
                    # But let's keep basic masks separately just in case user wants raw masks.

                    value_mask_img = (value_mask[:, :, np.newaxis] * 255).astype(np.uint8)
                    value_mask_img = np.repeat(value_mask_img, 3, axis=2)
                    policy_mask_img = (policy_mask[:, :, np.newaxis] * 255).astype(np.uint8)
                    policy_mask_img = np.repeat(policy_mask_img, 3, axis=2)

                    value_mask_images.append(value_mask_img)
                    policy_mask_images.append(policy_mask_img)

                    # Create Combined Figure: Original | Policy Attended | Value Attended
                    # Extract "Attended" panel from composites (2nd panel: index 1)
                    # Composite structure: [Image, Sep, Attended, Sep, Ignored, Sep, Heatmap]
                    w = image.shape[1]
                    s = 2
                    start_idx = w + s
                    end_idx = start_idx + w

                    val_att_panel = value_composite[:, start_idx:end_idx, :]
                    pol_att_panel = policy_composite[:, start_idx:end_idx, :]

                    sep = np.ones((image.shape[0], 2, 3), dtype=np.uint8) * 255
                    combined_figure = np.concatenate([image, sep, pol_att_panel, sep, val_att_panel], axis=1)

                    combined_images.append(combined_figure)

                    # Store attention images (masked images) separately if needed
                    # Or just use the composite as the main visualization.
                    # Since log_to_wandb expects these keys, we populate them.
                    # Previously these were single masked images. Now we can put the composite or the masked part.
                    # Let's populate them with the composite to be safe, or just the masked part if distinct.
                    # But wait, create_masked_composite DOES create the 3-panel view.
                    # Let's populate value_attention_images with value_composite to satisfy the loop index,
                    # but actually the user probably wants to see just the composite.

                    value_attention_images.append(value_composite)
                    policy_attention_images.append(policy_composite)

                    frames_generated += 1

        # Compute average sparsity (fraction of zero pixels)
        value_sparsity = np.mean(value_mask_zeros) / total_pixels if total_pixels else 0.0
        policy_sparsity = np.mean(policy_mask_zeros) / total_pixels if total_pixels else 0.0

        return {
            'value': value_images,
            'policy': policy_images,
            'combined': combined_images,
            # 'value_mask': value_mask_images,
            # 'policy_mask': policy_mask_images,
            'value_attention': value_attention_images,
            'policy_attention': policy_attention_images,
            'value_mask_sparsity': value_sparsity,
            'policy_mask_sparsity': policy_sparsity,
        }

    def log_to_wandb(self, global_step: int, prefix: str = "eval"):
        """
        Log visualization images to wandb.

        Args:
            global_step: Current training step
            prefix: Prefix for wandb keys
        """
        # Generate images (limit to 20 for logging to avoid spam)
        results = self.create_images(max_frames=20)

        if not results['value']:
            print(f"[GradientVis] No episodes collected for gradient visualization")
            return

        # Log aggregated metrics
        wandb.log({
            f"global_step": global_step,
            f"{prefix}/value_mask_sparsity": results['value_mask_sparsity'],
            f"{prefix}/policy_mask_sparsity": results['policy_mask_sparsity'],
        })

        # Log individual frames as images
        for i in range(len(results['value'])):
            wandb.log({
                # f"{prefix}/frame_{i}/value": wandb.Image(results['value'][i]),
                # f"{prefix}/frame_{i}/policy": wandb.Image(results['policy'][i]),
                f"{prefix}/frame_{i}/combined": wandb.Image(results['combined'][i]),
                # f"{prefix}/frame_{i}/value_attention": wandb.Image(results['value_attention'][i]),
                # f"{prefix}/frame_{i}/policy_attention": wandb.Image(results['policy_attention'][i]),
            })

        print(f"[GradientVis] Logged {len(self.episodes_data)} episodes (stats), "
              f"{len(results['value'])} frames (images) to wandb")
        print(f"[GradientVis] Mask sparsity - Value: {results['value_mask_sparsity']:.2%}, "
              f"Policy: {results['policy_mask_sparsity']:.2%}")


# =============================================================================
# High-level API for running gradient visualization during evaluation
# =============================================================================
#
# def run_gradient_visualization(
#     agent,
#     args,
#     global_step: int,
#     n_episodes: int = 5,
#     max_frames_per_episode: int = 500,
#     track: str = "both",
#     threshold: float = 0.01,
#     background_brightness: float = 0.0,
# ):
#     """
#     Run gradient visualization separately from main evaluation.
#
#     Args:
#         agent: PPO agent
#         args: Arguments (uses env_id, seed, obs_res, env_track_setting, deterministic_rollout)
#         global_step: Current training step for logging
#         n_episodes: Number of episodes to visualize (default: 5)
#         max_frames_per_episode: Max frames per episode (default: 500)
#         track: Which track to visualize - "train", "test", or "both"
#         threshold: Direct threshold in [0, 1] for attention masking (default: 0.01)
#         background_brightness: Brightness of non-attended regions (default: 0.0 = black)
#     """
#     from visual_scaling_drl.train.make_env import make_an_env
#
#     print(f"\n[GradientVis] Starting gradient visualization at step {global_step}")
#
#     if track in ["train", "both"]:
#         _run_gradient_vis_track(
#             agent, args, global_step, n_episodes, max_frames_per_episode,
#             full_distribution=False, threshold=threshold,
#             background_brightness=background_brightness, prefix="gradient_vis/train"
#         )
#
#     if track in ["test", "both"] and args.env_track_setting == "generalization":
#         _run_gradient_vis_track(
#             agent, args, global_step, n_episodes, max_frames_per_episode,
#             full_distribution=True, threshold=threshold,
#             background_brightness=background_brightness, prefix="gradient_vis/test"
#         )
#
#     print(f"[GradientVis] Done\n")
#
#
# def _run_gradient_vis_track(
#     agent, args, global_step, n_episodes, max_frames_per_episode,
#     full_distribution, threshold, background_brightness, prefix
# ):
#     """Run gradient visualization for a single track."""
#     from visual_scaling_drl.train.make_env import make_an_env
#
#     device = next(agent.parameters()).device
#
#     envs = make_an_env(
#         args, seed=args.seed, normalize_reward=False, obs_res=args.obs_res,
#         env_track_setting=args.env_track_setting, full_distribution=full_distribution
#     )
#
#     visualizer = GradientVisualizer(
#         agent=agent,
#         device=device,
#         n_episodes=n_episodes,
#         max_frames_per_episode=max_frames_per_episode,
#         threshold=threshold,
#         background_brightness=background_brightness,
#     )
#
#     # Track which env we're recording from
#     tracking_env_idx = 0
#
#     agent.eval()
#     next_obs, _ = envs.reset()
#     next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
#
#     step_count = 0
#
#     while visualizer.is_collecting():
#         # Collect gradient from tracked env
#         visualizer.step(next_obs[tracking_env_idx], done=False, env_idx=0)
#
#         # Get action
#         with torch.inference_mode():
#             action = agent.get_action(next_obs, deterministic=args.deterministic_rollout)
#
#         next_obs, _, terminated, truncated, info = envs.step(action.cpu().numpy())
#         next_obs = torch.tensor(next_obs, device=device, dtype=torch.float32)
#         step_count += 1
#
#         # Check if tracked env finished
#         episode_ended = False
#         if envs.env_type == "atari":
#             next_done = np.logical_or(terminated, truncated)
#             if next_done[tracking_env_idx] and info["lives"][tracking_env_idx] == 0:
#                 episode_ended = True
#         else:
#             if "_episode" in info.keys() and info["_episode"][tracking_env_idx]:
#                 episode_ended = True
#
#         if episode_ended:
#             visualizer.step(next_obs[tracking_env_idx], done=True, env_idx=0)
#             print(f"[GradientVis] Episode {visualizer.episodes_collected}/{n_episodes} saved "
#                   f"({len(visualizer.episodes_data[-1]) if visualizer.episodes_data else 0} frames)")
#
#     agent.train()
#     envs.close()
#
#     visualizer.log_to_wandb(global_step, prefix=prefix)