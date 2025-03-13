from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import spconv.pytorch as sp

from ..modules.utils import convert_module_to_f16, convert_module_to_f32
from .structured_latent_flow import SLatFlowModel


class CameraAwareSparseBlock(nn.Module):
    """Camera-aware sparse residual block for the SLAT model."""
    def __init__(self, in_channels, out_channels, camera_channels=256):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Main convolution path
        self.conv1 = sp.SubMConv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.conv2 = sp.SubMConv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.norm2 = nn.BatchNorm1d(out_channels)
        
        # Camera conditioning path
        self.camera_proj = nn.Linear(camera_channels, out_channels * 2)  # scale and shift
        
        # Residual connection
        if in_channels != out_channels:
            self.residual = sp.SubMConv3d(in_channels, out_channels, kernel_size=1, bias=True)
        else:
            self.residual = nn.Identity()
    
    def forward(self, x: sp.SparseConvTensor, camera_cond: torch.Tensor = None) -> sp.SparseConvTensor:
        """Forward pass with optional camera conditioning
        
        Args:
            x (sp.SparseConvTensor): Input sparse tensor
            camera_cond (torch.Tensor, optional): Camera conditioning features [B, C]
            
        Returns:
            sp.SparseConvTensor: Output sparse tensor
        """
        identity = x
        
        # First conv block
        out = self.conv1(x)
        out_feats = out.features
        out_feats = self.norm1(out_feats)
        out_feats = F.relu(out_feats)
        out = out.replace_feature(out_feats)
        
        # Second conv block
        out = self.conv2(out)
        out_feats = out.features
        out_feats = self.norm2(out_feats)
        
        # Apply camera conditioning if available
        if camera_cond is not None:
            # Project camera condition to scale and shift
            cam_params = self.camera_proj(camera_cond)  # [B, out_channels*2]
            scale, shift = torch.chunk(cam_params, 2, dim=1)  # [B, out_channels] each
            
            # Get batch indices from sparse tensor
            batch_indices = out.indices[:, 0]
            
            # Apply scale and shift based on batch index
            batch_scale = scale[batch_indices]  # [N, out_channels]
            batch_shift = shift[batch_indices]  # [N, out_channels]
            
            # Modulate features
            out_feats = out_feats * (1 + batch_scale) + batch_shift
            
        # Apply ReLU
        out_feats = F.relu(out_feats)
        out = out.replace_feature(out_feats)
        
        # Apply residual connection
        identity = self.residual(identity) if not isinstance(self.residual, nn.Identity) else identity
        out_feats = out.features + identity.features
        out = out.replace_feature(out_feats)
        
        return out


class UncertaintyModule(nn.Module):
    """Estimates uncertainty in generated structured latents."""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        # MLP for uncertainty estimation
        self.uncertainty_net = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(channels // 2, channels // 4),
            nn.LeakyReLU(0.2),
            nn.Linear(channels // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features):
        """Estimate uncertainty from features
        
        Args:
            features (torch.Tensor): Input features [N, C]
            
        Returns:
            torch.Tensor: Uncertainty values [N, 1]
        """
        return self.uncertainty_net(features)


class EnhancedSLatFlowModel(SLatFlowModel):
    """Enhanced Structured Latent Flow Model with camera awareness and uncertainty."""
    def __init__(
        self,
        in_channels: int,
        cond_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_input_blocks: int = 3,
        num_output_blocks: int = 3,
        num_bottleneck_blocks: int = 6,
        skip_connections: bool = True,
        use_viewdir: bool = False,
        camera_channels: int = 256,
        use_uncertainty: bool = True,
        use_fp16: bool = False,
    ):
        super().__init__(
            in_channels=in_channels,
            cond_channels=cond_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_input_blocks=num_input_blocks,
            num_output_blocks=num_output_blocks,
            num_bottleneck_blocks=num_bottleneck_blocks,
            skip_connections=skip_connections,
            use_viewdir=use_viewdir,
            use_fp16=use_fp16,
        )
        
        # Replace standard blocks with camera-aware blocks
        # Input blocks
        self.input_blocks = nn.ModuleList([
            CameraAwareSparseBlock(in_channels if i == 0 else hidden_channels, hidden_channels, camera_channels)
            for i in range(num_input_blocks)
        ])
        
        # Bottleneck blocks
        self.bottleneck_blocks = nn.ModuleList([
            CameraAwareSparseBlock(hidden_channels, hidden_channels, camera_channels)
            for _ in range(num_bottleneck_blocks)
        ])
        
        # Output blocks
        self.output_blocks = nn.ModuleList([
            CameraAwareSparseBlock(
                2 * hidden_channels if skip_connections and i > 0 else hidden_channels, 
                hidden_channels if i < num_output_blocks - 1 else out_channels,
                camera_channels
            )
            for i in range(num_output_blocks)
        ])
        
        # Camera conditioning module
        self.camera_embedding = nn.Sequential(
            nn.Linear(16, camera_channels // 2),  # 4x4 matrix flattened
            nn.LeakyReLU(0.2),
            nn.Linear(camera_channels // 2, camera_channels),
            nn.LeakyReLU(0.2)
        )
        
        # Multi-camera fusion
        self.camera_fusion = nn.Sequential(
            nn.Linear(camera_channels * 4, camera_channels * 2),  # 4 views
            nn.LeakyReLU(0.2),
            nn.Linear(camera_channels * 2, camera_channels)
        )
        
        # Uncertainty estimation
        self.use_uncertainty = use_uncertainty
        if use_uncertainty:
            self.uncertainty_estimator = UncertaintyModule(hidden_channels)
    
    def process_camera_params(self, camera_params):
        """Process camera parameters into conditioning vectors
        
        Args:
            camera_params (List[Dict]): List of camera parameter dictionaries
                
        Returns:
            torch.Tensor: Camera conditioning embedding
        """
        # Extract up to 4 camera extrinsics
        cam_tensors = []
        for params in camera_params[:4]:  # Limit to 4 cameras
            # Convert to tensor if needed
            if isinstance(params['extrinsic'], np.ndarray):
                extrinsic = torch.tensor(params['extrinsic'], 
                                         dtype=torch.float32).to(next(self.parameters()).device)
            else:
                extrinsic = params['extrinsic']
            
            # Flatten
            cam_tensors.append(extrinsic.flatten())
        
        # Pad if fewer than 4 cameras
        while len(cam_tensors) < 4:
            padding = torch.eye(4, dtype=torch.float32).flatten().to(next(self.parameters()).device)
            cam_tensors.append(padding)
        
        # Process each camera individually
        cam_embeddings = [self.camera_embedding(cam) for cam in cam_tensors]
        
        # Concatenate and fuse
        cat_embeddings = torch.cat(cam_embeddings, dim=0)
        fused_embedding = self.camera_fusion(cat_embeddings)
        
        return fused_embedding
    
    def forward(self, x: sp.SparseTensor, t: torch.Tensor, cond: torch.Tensor, camera_params=None) -> sp.SparseTensor:
        """Forward pass with camera conditioning
        
        Args:
            x (sp.SparseTensor): Input sparse tensor
            t (torch.Tensor): Timestep embedding
            cond (torch.Tensor): Conditioning from image encoder
            camera_params (List[Dict], optional): Camera parameters for multiple views
            
        Returns:
            sp.SparseTensor: Output sparse tensor
            torch.Tensor: Uncertainty values (optional)
        """
        # Process input
        h = self.input_layer(x).type(self.dtype)
        t_emb = self.t_embedder(t)
        if self.conditional:
            t_emb = self.adaLN_modulation(t_emb)
        t_emb = t_emb.type(self.dtype)
        
        # Process camera parameters if provided
        cam_cond = None
        if camera_params is not None:
            cam_cond = self.process_camera_params(camera_params).type(self.dtype)
        
        # Process with input blocks
        skip_features = []
        for block in self.input_blocks:
            h = block(h, cam_cond)
            skip_features.append(h)
        
        # Process with bottleneck blocks
        for block in self.bottleneck_blocks:
            h = block(h, cam_cond)
        
        # Save features for uncertainty estimation
        uncertainty_features = h.features if self.use_uncertainty else None
        
        # Process with output blocks
        for i, block in enumerate(self.output_blocks):
            if self.skip_connections and i > 0:
                # Combine skip features with current features
                skip_h = skip_features[-(i+1)]
                h_inds, skip_inds = h.indices, skip_h.indices
                
                # Find common indices
                h_coords = torch.cat([h_inds[:, 0:1], h.grid_sizes[0] * h_inds[:, 1:]], dim=1)
                skip_coords = torch.cat([skip_inds[:, 0:1], h.grid_sizes[0] * skip_inds[:, 1:]], dim=1)
                
                # Convert to hashable format for set operations
                h_coords_tuple = [tuple(h_coord.cpu().numpy()) for h_coord in h_coords]
                skip_coords_tuple = [tuple(skip_coord.cpu().numpy()) for skip_coord in skip_coords]
                
                # Find the intersection
                common_coords = set(h_coords_tuple).intersection(set(skip_coords_tuple))
                
                # Create masks for common indices
                h_mask = torch.tensor([coord in common_coords for coord in h_coords_tuple], 
                                    device=h.features.device, dtype=torch.bool)
                skip_mask = torch.tensor([coord in common_coords for coord in skip_coords_tuple], 
                                        device=skip_h.features.device, dtype=torch.bool)
                
                # Extract the features at common indices
                h_common_feats = h.features[h_mask]
                skip_common_feats = skip_h.features[skip_mask]
                
                # Concatenate the features
                concat_feats = torch.cat([h_common_feats, skip_common_feats], dim=1)
                
                # Update the features at common indices
                h_new_feats = h.features.clone()
                h_new_feats[h_mask] = concat_feats
                h = h.replace_feature(h_new_feats)
            
            # Apply block
            h = block(h, cam_cond)
        
        # Calculate uncertainty if requested
        uncertainty = None
        if self.use_uncertainty and uncertainty_features is not None:
            uncertainty = self.uncertainty_estimator(uncertainty_features)
        
        # Convert back to input data type
        if h.features.dtype != x.features.dtype:
            h = h.replace_feature(h.features.type(x.features.dtype))
        
        if uncertainty is not None:
            return h, uncertainty
        return h
