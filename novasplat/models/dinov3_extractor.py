"""DINOv3 feature extractor module."""

import torch


class DINOv3Extractor:
    """
    Wrapper for DINOv3 model feature extraction.

    Extracts intermediate layer features from a DINOv3 model for a batch of images.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        n_layers: int,
        num_output_layers: int = 1,
        device=None,
    ):
        """
        Initialize the DINOv3Extractor.

        Args:
            model (torch.nn.Module): The DINOv3 model.
            n_layers (int): Number of layers in the model.
            num_output_layers (int): Number of final transformer layers to
                concatenate for each output descriptor.
            device (optional): Device to move the model to (e.g., 'cuda' or 'cpu').
        """
        self.model = model.eval().to(device)
        self.n_layers = n_layers
        if num_output_layers <= 0:
            raise ValueError(f"num_output_layers must be > 0, got {num_output_layers}")
        self.num_output_layers = min(num_output_layers, n_layers)

    @torch.inference_mode()
    def extract(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract features from one or more final intermediate layers for a batch of images.

        Args:
            images (torch.Tensor): Batch of images, shape [B, C, H, W].

        Returns:
            torch.Tensor: Concatenated features from the final intermediate
            layers, shape [B, C * num_output_layers, H, W].
        """
        layer_ids = list(range(self.n_layers - self.num_output_layers, self.n_layers))
        feats = self.model.get_intermediate_layers(
            images,
            n=layer_ids,
            reshape=True,
            norm=True,
        )
        return torch.cat(feats, dim=1)
