from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..modules.utils import convert_module_to_f16, convert_module_to_f32
from ..modules.transformer import AbsolutePositionEmbedder, ModulatedTransformerCrossBlock
from ..modules.spatial import patchify, unpatchify
from .sparse_structure_flow import SparseStructureFlowModel, TimestepEmbedder


class CameraConditionedFlow(nn.Module):
    """Conditioning module that incorporates camera parameters into the flow model."""
    def __init__(self, model_channels, num_cameras=4):
        super().__init__()
        self.model_channels = model_channels
        self.num_cameras = num_cameras
        
        # Camera parameter embedding
        self.camera_embedder = nn.Sequential(
            nn.Linear(16, model_channels // 2),  # 4x4 extrinsic matrix flattened
            nn.SiLU(),
            nn.Linear(model_channels // 2, model_channels)
        )
        
        # Combine camera embeddings
        self.camera_fusion = nn.MultiheadAttention(
            embed_dim=model_channels,
            num_heads=8,
            batch_first=True
        )
        
        # Final projection
        self.projector = nn.Sequential(
            nn.LayerNorm(model_channels),
            nn.Linear(model_channels, model_channels),
            nn.SiLU(),
            nn.Linear(model_channels, model_channels)
        )
        
    def forward(self, camera_params):
        """Process camera parameters into conditioning vector
        
        Args:
            camera_params (List[Dict]): List of camera parameter dictionaries
                Each containing 'extrinsic' matrix
                
        Returns:
            torch.Tensor: Camera conditioning embedding
        """
        # Extract extrinsic matrices
        extrinsics = []
        for params in camera_params[:self.num_cameras]:
            # Convert to tensor if needed
            if isinstance(params['extrinsic'], np.ndarray):
                extrinsic = torch.tensor(params['extrinsic'], 
                                         dtype=torch.float32).to(next(self.parameters()).device)
            else:
                extrinsic = params['extrinsic']
            
            # Flatten and embed
            extrinsic_flat = extrinsic.flatten()
            extrinsics.append(extrinsic_flat)
            
        # Pad if we have fewer than expected cameras
        while len(extrinsics) < self.num_cameras:
            # Create identity matrix as padding
            padding = torch.eye(4, dtype=torch.float32).flatten().to(next(self.parameters()).device)
            extrinsics.append(padding)
            
        # Stack and embed
        extrinsics = torch.stack(extrinsics)  # [num_cameras, 16]
        embeddings = self.camera_embedder(extrinsics)  # [num_cameras, model_channels]
        
        # Multi-head attention to fuse camera information
        attn_output, _ = self.camera_fusion(
            embeddings.unsqueeze(0),  # [1, num_cameras, model_channels]
            embeddings.unsqueeze(0),
            embeddings.unsqueeze(0)
        )
        
        # Mean pooling over cameras
        fused_embedding = attn_output.mean(dim=1)  # [1, model_channels]
        
        # Final projection
        return self.projector(fused_embedding)  # [1, model_channels]


class EnhancedSparseStructureFlowModel(SparseStructureFlowModel):
    """Enhanced sparse structure flow model with multi-view and camera awareness."""
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        out_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        patch_size: int = 2,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
    ):
        super().__init__(
            resolution=resolution,
            in_channels=in_channels,
            model_channels=model_channels,
            cond_channels=cond_channels,
            out_channels=out_channels,
            num_blocks=num_blocks,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            mlp_ratio=mlp_ratio,
            patch_size=patch_size,
            pe_mode=pe_mode,
            use_fp16=use_fp16,
            use_checkpoint=use_checkpoint,
            share_mod=share_mod,
            qk_rms_norm=qk_rms_norm,
            qk_rms_norm_cross=qk_rms_norm_cross,
        )
        
        # Additional components for camera conditioning
        self.camera_conditioning = CameraConditionedFlow(model_channels)
        
        # Uncertainty estimation module
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(model_channels, model_channels // 2),
            nn.SiLU(),
            nn.Linear(model_channels // 2, 1),
            nn.Sigmoid()
        )
        
        # Multi-resolution blocks
        self.multi_res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(model_channels),
                nn.Linear(model_channels, model_channels),
                nn.SiLU(),
                nn.Linear(model_channels, model_channels)
            ) for _ in range(3)  # 3 resolution levels
        ])
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor, camera_params=None) -> torch.Tensor:
        """Forward pass with camera conditioning
        
        Args:
            x (torch.Tensor): Input tensor [B, C, D, H, W]
            t (torch.Tensor): Timestep embedding [B]
            cond (torch.Tensor): Conditioning from image encoder [B, T, D]
            camera_params (List[Dict], optional): Camera parameters for multiple views
            
        Returns:
            torch.Tensor: Output tensor
            torch.Tensor: Uncertainty estimate
        """
        # Standard SparseStructureFlowModel forward
        assert [*x.shape] == [x.shape[0], self.in_channels, *[self.resolution] * 3], \
                f"Input shape mismatch, got {x.shape}, expected {[x.shape[0], self.in_channels, *[self.resolution] * 3]}"

        h = patchify(x, self.patch_size)
        h = h.view(*h.shape[:2], -1).permute(0, 2, 1).contiguous()

        h = self.input_layer(h)
        
        # Add positional embedding
        h = h + self.pos_emb[None]
        
        # Get timestep embedding
        t_emb = self.t_embedder(t)
        
        # Add camera conditioning if available
        cam_cond = None
        if camera_params is not None:
            cam_cond = self.camera_conditioning(camera_params)
            # Add to timestep embedding
            t_emb = t_emb + cam_cond
            
        # Process timestep embedding
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)
        t_emb = t_emb.type(self.dtype)
        h = h.type(self.dtype)
        cond = cond.type(self.dtype)
        
        # Process with transformer blocks
        # Process at multiple resolutions
        multi_res_features = []
        
        for i, block in enumerate(self.blocks):
            h = block(h, t_emb, cond)
            
            # Process at different resolutions periodically
            if (i + 1) % (len(self.blocks) // 3) == 0:
                res_idx = (i + 1) // (len(self.blocks) // 3) - 1
                multi_res_h = h.type(torch.float32)
                multi_res_h = self.multi_res_blocks[res_idx](multi_res_h)
                multi_res_features.append(multi_res_h)
                h = h + multi_res_h.type(self.dtype)
        
        # Convert back to float32 for output
        h = h.type(x.dtype)
        h = F.layer_norm(h, h.shape[-1:])
        
        # Estimate uncertainty
        uncertainty = None
        if len(multi_res_features) > 0:
            # Combine multi-resolution features
            combined_features = torch.stack(multi_res_features).mean(dim=0)
            uncertainty = self.uncertainty_estimator(combined_features)
        
        # Generate output
        h = self.out_layer(h)

        h = h.permute(0, 2, 1).view(h.shape[0], h.shape[2], *[self.resolution // self.patch_size] * 3)
        h = unpatchify(h, self.patch_size).contiguous()

        return h, uncertainty
