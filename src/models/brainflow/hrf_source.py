import torch
import torch.nn as nn

class AECNN_HRF_Source(nn.Module):
    """
    Condition-dependent Source Generator using Biological HRF Prior.
    Compresses temporal context into neural events and filters through an HRF conv layer
    to create a biological base distribution (mu_phi) and variance (sigma_phi).
    """
    def __init__(self, context_dim: int, latent_dim: int, hrf_kernel_size: int = 12):
        super().__init__()
        # Step 1: Neural Event Extractor
        self.neural_event_net = nn.Sequential(
            nn.Conv1d(context_dim, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(512, latent_dim, kernel_size=3, padding=1),
            nn.Sigmoid()  # Soft binarization to simulate neural firing
        )
        
        # Step 2: HRF Convolutional Filter
        # Linear Conv1D without activation simulating hemodynamic convolution
        self.hrf_filter = nn.Conv1d(
            latent_dim, latent_dim, 
            kernel_size=hrf_kernel_size, 
            padding='same', 
            groups=latent_dim
        )
        
        # Sigma Predictor (Variance)
        self.sigma_net = nn.Sequential(
            nn.Linear(context_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Softplus()  # Ensure strictly positive variance
        )

    def forward(self, context_sequence: torch.Tensor, context_pooled: torch.Tensor):
        # context_sequence: [B, C, T]
        # context_pooled: [B, C]
        
        neural_events = self.neural_event_net(context_sequence) # [B, latent_dim, T]
        mu_phi = self.hrf_filter(neural_events)                # [B, latent_dim, T]
        mu_phi = mu_phi.transpose(1, 2)                        # [B, T, latent_dim]
        
        sigma_phi = self.sigma_net(context_pooled).unsqueeze(1) # [B, 1, 1]
        
        return mu_phi, sigma_phi
