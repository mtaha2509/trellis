from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CameraParameterEmbedding(nn.Module):
    """Embeds camera parameters (extrinsic and intrinsic) into a feature vector.
    
    This embedding can be concatenated with image features to make them camera-aware.
    """
    def __init__(self, embedding_dim=256):
        super().__init__()
        # For extrinsic parameters (rotation and translation)
        self.rotation_embedding = nn.Sequential(
            nn.Linear(9, embedding_dim // 2),  # 3x3 rotation matrix flattened
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, embedding_dim // 2)
        )
        
        self.translation_embedding = nn.Sequential(
            nn.Linear(3, embedding_dim // 4),  # 3D translation vector
            nn.ReLU(),
            nn.Linear(embedding_dim // 4, embedding_dim // 4)
        )
        
        # For intrinsic parameters
        self.intrinsic_embedding = nn.Sequential(
            nn.Linear(4, embedding_dim // 4),  # fx, fy, cx, cy
            nn.ReLU(),
            nn.Linear(embedding_dim // 4, embedding_dim // 4)
        )
        
        # Final projector
        self.projector = nn.Linear(embedding_dim, embedding_dim)
        
    def forward(self, camera_params):
        """Convert camera parameters to embedding
        
        Args:
            camera_params (dict): Dictionary containing 'extrinsic' and 'intrinsic' matrices
            
        Returns:
            torch.Tensor: Camera parameter embedding
        """
        # Extract extrinsic parameters
        extrinsic = camera_params['extrinsic']
        rotation = extrinsic[:3, :3].flatten()  # 3x3 -> 9
        translation = extrinsic[:3, 3]  # 3x1 -> 3
        
        # Extract intrinsic parameters
        intrinsic = camera_params['intrinsic']
        # Extract focal lengths and principal point
        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]
        intrinsic_params = torch.tensor([fx, fy, cx, cy], device=rotation.device)
        
        # Generate embeddings
        rot_emb = self.rotation_embedding(rotation)
        trans_emb = self.translation_embedding(translation)
        intr_emb = self.intrinsic_embedding(intrinsic_params)
        
        # Concatenate and project
        combined_emb = torch.cat([rot_emb, trans_emb, intr_emb], dim=0)
        return self.projector(combined_emb)


class CameraAttentionBlock(nn.Module):
    """Self-attention block that incorporates camera parameter information."""
    def __init__(self, feature_dim, num_heads=8, dropout=0.1):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization and feedforward
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        
        self.feedforward = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Linear(feature_dim * 4, feature_dim),
            nn.Dropout(dropout)
        )
        
    def with_pos_embed(self, tensor, pos=None):
        """Add positional embedding to tensor if provided"""
        return tensor if pos is None else tensor + pos
        
    def forward(self, features, cam_embeddings=None):
        """Apply attention with camera embedding as positional encoding
        
        Args:
            features (torch.Tensor): Features from multiple views [B*V, T, D]
            cam_embeddings (torch.Tensor, optional): Camera parameter embeddings [B*V, D]
            
        Returns:
            torch.Tensor: Updated features with cross-view attention
        """
        # Reshape cam_embeddings to add as positional embeddings
        if cam_embeddings is not None:
            # Expand cam_embeddings from [B*V, D] to [B*V, T, D]
            cam_embeddings = cam_embeddings.unsqueeze(1).expand(-1, features.shape[1], -1)
        
        # Self-attention with camera embeddings as positional encoding
        query = self.with_pos_embed(features, cam_embeddings)
        key = self.with_pos_embed(features, cam_embeddings)
        value = features
        
        # Apply multi-head attention
        features_norm = self.norm1(features)
        attn_output, _ = self.attention(query, key, value)
        features = features + attn_output
        
        # Feedforward
        features = features + self.feedforward(self.norm2(features))
        
        return features


class CrossViewCorrelation(nn.Module):
    """Computes correlations between features from different views based on camera geometry.
    
    This module helps establish correspondences between features across different views
    by leveraging camera parameters to compute geometric relationships.
    """
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Projection layers to create keys and queries for attention
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        
        # Camera projection layer
        self.camera_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, feature_dim)
        )
    
    def compute_fundamental_matrix(self, camera_i, camera_j):
        """Compute the fundamental matrix between two views"""
        # Convert numpy arrays to tensors if needed
        if isinstance(camera_i['extrinsic'], np.ndarray):
            camera_i['extrinsic'] = torch.from_numpy(camera_i['extrinsic']).float()
        if isinstance(camera_i['intrinsic'], np.ndarray):
            camera_i['intrinsic'] = torch.from_numpy(camera_i['intrinsic']).float()
        if isinstance(camera_j['extrinsic'], np.ndarray):
            camera_j['extrinsic'] = torch.from_numpy(camera_j['extrinsic']).float()
        if isinstance(camera_j['intrinsic'], np.ndarray):
            camera_j['intrinsic'] = torch.from_numpy(camera_j['intrinsic']).float()
        
        # Get device
        device = camera_i['extrinsic'].device
        
        # Extract camera matrices
        K_i = camera_i['intrinsic']
        R_i = camera_i['extrinsic'][:3, :3]
        t_i = camera_i['extrinsic'][:3, 3]
        
        K_j = camera_j['intrinsic']
        R_j = camera_j['extrinsic'][:3, :3]
        t_j = camera_j['extrinsic'][:3, 3]
        
        # Compute relative rotation and translation
        R_rel = torch.matmul(R_j, R_i.transpose(-2, -1))
        t_rel = t_j - torch.matmul(R_rel, t_i)
        
        # Create skew-symmetric matrix from translation
        t_skew = torch.zeros(3, 3, device=device)
        t_skew[0, 1] = -t_rel[2]
        t_skew[0, 2] = t_rel[1]
        t_skew[1, 0] = t_rel[2]
        t_skew[1, 2] = -t_rel[0]
        t_skew[2, 0] = -t_rel[1]
        t_skew[2, 1] = t_rel[0]
        
        # Compute essential matrix: E = t_skew * R_rel
        E = torch.matmul(t_skew, R_rel)
        
        # Compute fundamental matrix: F = K_j^-T * E * K_i^-1
        K_i_inv = torch.inverse(K_i)
        K_j_inv_t = torch.inverse(K_j).transpose(-2, -1)
        F = torch.matmul(K_j_inv_t, torch.matmul(E, K_i_inv))
        
        return F
    
    def compute_epipolar_attention(self, features_i, features_j, F, cam_embed_i, cam_embed_j):
        """Compute attention weights based on epipolar geometry"""
        batch_size, seq_len, _ = features_i.shape
        
        # Project features with camera awareness
        q = self.query_proj(features_i + cam_embed_i)  # [B, T, D]
        k = self.key_proj(features_j + cam_embed_j)    # [B, T, D]
        
        # Compute attention scores
        attention = torch.matmul(q, k.transpose(-2, -1)) / (self.feature_dim ** 0.5)  # [B, T, T]
        
        # Apply epipolar constraint as an attention mask
        # This is a simplified version - a full implementation would compute
        # epipolar lines for each point and mask based on distance to line
        epipolar_weight = torch.softmax(attention, dim=-1)
        
        return epipolar_weight
    
    def forward(self, features_list, camera_params):
        """Compute cross-view correlations
        
        Args:
            features_list (List[torch.Tensor]): List of features from each view [B, T, D]
            camera_params (List[Dict]): List of camera parameters for each view
            
        Returns:
            List[torch.Tensor]: List of enhanced features with cross-view correlations
        """
        device = features_list[0].device
        num_views = len(features_list)
        batch_size = features_list[0].shape[0]
        seq_len = features_list[0].shape[1]
        
        # Create camera embeddings for each view
        cam_embeddings = []
        for params in camera_params:
            # Convert numpy arrays to tensors if needed
            if isinstance(params['extrinsic'], np.ndarray):
                params['extrinsic'] = torch.from_numpy(params['extrinsic']).float().to(device)
            if isinstance(params['intrinsic'], np.ndarray):
                params['intrinsic'] = torch.from_numpy(params['intrinsic']).float().to(device)
            
            # Create camera embedding
            cam_emb = self.camera_proj(torch.cat([
                params['extrinsic'][:3, :3].flatten(),
                params['extrinsic'][:3, 3],
                params['intrinsic'][0, 0].unsqueeze(0),  # fx
                params['intrinsic'][1, 1].unsqueeze(0),  # fy
                params['intrinsic'][0, 2].unsqueeze(0),  # cx
                params['intrinsic'][1, 2].unsqueeze(0)   # cy
            ])).unsqueeze(0).expand(batch_size, seq_len, -1)  # [B, T, D]
            
            cam_embeddings.append(cam_emb)
        
        # Enhanced features with cross-view correlations
        enhanced_features = []
        
        # For each view, compute attention from all other views
        for i in range(num_views):
            # Start with original features
            view_features = features_list[i].clone()  # [B, T, D]
            
            # Add information from other views
            for j in range(num_views):
                if i == j:
                    continue
                
                # Compute fundamental matrix between views
                F = self.compute_fundamental_matrix(camera_params[i], camera_params[j])
                
                # Compute attention weights
                attn_weights = self.compute_epipolar_attention(
                    features_list[i], features_list[j],
                    F, cam_embeddings[i], cam_embeddings[j]
                )  # [B, T, T]
                
                # Apply attention to get features from view j
                attended_features = torch.matmul(attn_weights, features_list[j])  # [B, T, D]
                
                # Add to current view features
                view_features = view_features + 0.2 * attended_features  # scaled addition
            
            enhanced_features.append(view_features)
        
        return enhanced_features


class CameraAwareFeatureFusion(nn.Module):
    """Fuses features from multiple views using camera parameters.
    
    This module takes features from multiple views and their corresponding camera
    parameters, and fuses them into a single set of features that are camera-aware.
    """
    def __init__(self, feature_dim=768, hidden_dim=1024, num_heads=8, num_layers=3):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # Camera parameter embedding
        self.camera_embedding = CameraParameterEmbedding(embedding_dim=feature_dim)
        
        # Initial feature projection
        self.feature_projector = nn.Linear(feature_dim, hidden_dim)
        
        # Cross-view attention blocks
        self.attention_blocks = nn.ModuleList([
            CameraAttentionBlock(hidden_dim, num_heads=num_heads)
            for _ in range(num_layers)
        ])
        
        # Cross-view correlation module
        self.correlation_module = CrossViewCorrelation(hidden_dim)
        
        # Output projection
        self.output_projector = nn.Linear(hidden_dim, feature_dim)
        
    def forward(self, view_features, camera_params):
        """Fuse features from multiple views with camera awareness
        
        Args:
            view_features (List[torch.Tensor]): List of features from each view
                Each tensor has shape [B, T, D] where T is the sequence length
            camera_params (List[Dict]): List of camera parameters for each view
            
        Returns:
            torch.Tensor: Fused features with camera awareness [B, T, D]
        """
        batch_size = view_features[0].shape[0]
        num_views = len(view_features)
        device = view_features[0].device
        
        # Process camera parameters
        cam_embeddings = []
        for params in camera_params:
            # Convert numpy arrays to tensors if needed
            if isinstance(params['extrinsic'], np.ndarray):
                params['extrinsic'] = torch.from_numpy(params['extrinsic']).float().to(device)
            if isinstance(params['intrinsic'], np.ndarray):
                params['intrinsic'] = torch.from_numpy(params['intrinsic']).float().to(device)
                
            emb = self.camera_embedding(params)
            cam_embeddings.append(emb)
            
        cam_embeddings = torch.stack(cam_embeddings)  # [V, D]
        
        # Process features from each view
        projected_features = []
        for view_idx, features in enumerate(view_features):
            # Project features to hidden dimension
            proj_feats = self.feature_projector(features)  # [B, T, H]
            projected_features.append(proj_feats)
            
        # Apply cross-view correlation
        correlated_features = self.correlation_module(projected_features, camera_params)
        
        # Stack all features from all views
        # [B, V, T, H] -> [B*V, T, H]
        stacked_features = torch.stack(correlated_features, dim=1)
        original_shape = stacked_features.shape
        stacked_features = stacked_features.view(batch_size * num_views, *stacked_features.shape[2:])
        
        # Repeat camera embeddings for each batch
        # [V, D] -> [B*V, D]
        stacked_cam_embeddings = cam_embeddings.repeat(batch_size, 1)
        
        # Apply attention blocks with camera awareness
        for block in self.attention_blocks:
            stacked_features = block(stacked_features, stacked_cam_embeddings)
            
        # Reshape back and average across views
        # [B*V, T, H] -> [B, V, T, H] -> [B, T, H]
        fused_features = stacked_features.view(*original_shape)
        fused_features = fused_features.mean(dim=1)  # Average across views
        
        # Project back to original dimension
        fused_features = self.output_projector(fused_features)
        
        return fused_features


class RaySpaceFusion(nn.Module):
    """Projects features into a unified 3D ray space based on camera parameters.
    
    This is a more advanced fusion method that explicitly models 3D space.
    """
    def __init__(self, feature_dim=768, grid_resolution=32, depth_resolution=32):
        super().__init__()
        self.feature_dim = feature_dim
        self.grid_resolution = grid_resolution
        self.depth_resolution = depth_resolution
        
        # Feature projector
        self.feature_projector = nn.Conv2d(
            feature_dim, feature_dim,
            kernel_size=1, stride=1, padding=0
        )
        
        # 3D volume feature extractor
        self.volume_encoder = nn.Sequential(
            nn.Conv3d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(feature_dim, feature_dim, kernel_size=3, padding=1),
        )
        
        # Visibility network to predict occlusion
        self.visibility_net = nn.Sequential(
            nn.Conv3d(feature_dim, feature_dim//2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(feature_dim//2, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
    def create_3d_grid(self, batch_size, device):
        """Create a 3D grid of points in world space"""
        # Create normalized grid coordinates from -1 to 1
        grid_x = torch.linspace(-1, 1, self.grid_resolution, device=device)
        grid_y = torch.linspace(-1, 1, self.grid_resolution, device=device)
        grid_z = torch.linspace(-1, 1, self.depth_resolution, device=device)
        
        # Create 3D grid
        grid_x, grid_y, grid_z = torch.meshgrid(grid_x, grid_y, grid_z)
        
        # Reshape to [B, 3, D, H, W]
        grid_points = torch.stack([grid_x, grid_y, grid_z], dim=0).unsqueeze(0)
        grid_points = grid_points.expand(batch_size, -1, -1, -1, -1)
        
        # Reshape to [B, D*H*W, 3]
        points_3d = grid_points.permute(0, 2, 3, 4, 1).reshape(batch_size, -1, 3)
        
        return points_3d
        
    def project_points_to_image(self, points_3d, camera_params):
        """Project 3D points to 2D image coordinates
        
        Args:
            points_3d (torch.Tensor): 3D points in world space [B, N, 3]
            camera_params (Dict): Camera parameters
            
        Returns:
            torch.Tensor: 2D coordinates in image space [-1,1] range [B, N, 2]
        """
        batch_size, num_points, _ = points_3d.shape
        device = points_3d.device
        
        # Convert numpy arrays to tensors if needed
        if isinstance(camera_params['extrinsic'], np.ndarray):
            camera_params['extrinsic'] = torch.from_numpy(camera_params['extrinsic']).float().to(device)
        if isinstance(camera_params['intrinsic'], np.ndarray):
            camera_params['intrinsic'] = torch.from_numpy(camera_params['intrinsic']).float().to(device)
        
        # Extract camera parameters
        R = camera_params['extrinsic'][:3, :3]  # [3, 3]
        t = camera_params['extrinsic'][:3, 3]  # [3]
        K = camera_params['intrinsic']  # [3, 3]
        
        # Transform points from world to camera space
        # [B, N, 3] -> [B, N, 3]
        points_cam = torch.bmm(points_3d, R.transpose(0, 1).unsqueeze(0).expand(batch_size, -1, -1))
        points_cam = points_cam + t.unsqueeze(0).unsqueeze(1).expand(batch_size, num_points, -1)
        
        # Project to image space
        # [B, N, 3] -> [B, N, 3]
        points_img = torch.bmm(points_cam, K.transpose(0, 1).unsqueeze(0).expand(batch_size, -1, -1))
        
        # Convert to homogeneous coordinates
        # [B, N, 3] -> [B, N, 2]
        points_img = points_img[:, :, :2] / (points_img[:, :, 2:] + 1e-6)
        
        # Normalize to [-1, 1] for grid_sample
        height, width = camera_params.get('height', 256), camera_params.get('width', 256)
        points_img[:, :, 0] = (points_img[:, :, 0] / width) * 2 - 1
        points_img[:, :, 1] = (points_img[:, :, 1] / height) * 2 - 1
        
        return points_img
    
    def project_features_to_3d(self, features, depths, camera_params):
        """Project 2D features to 3D space using depth values and camera parameters"""
        batch_size = features.shape[0]
        device = features.device
        
        # Create 3D grid points
        points_3d = self.create_3d_grid(batch_size, device)  # [B, D*H*W, 3]
        
        # Project 3D points to 2D image coordinates
        points_2d = self.project_points_to_image(points_3d, camera_params)  # [B, D*H*W, 2]
        
        # Reshape for grid_sample: [B, D*H*W, 2] -> [B, D, H, W, 2]
        points_2d = points_2d.view(
            batch_size, self.depth_resolution, self.grid_resolution, self.grid_resolution, 2
        )
        
        # Apply grid_sample to sample features at 2D locations
        # Need to reshape features first if they're not in the right format
        if len(features.shape) == 3:  # [B, T, D]
            seq_len, feat_dim = features.shape[1], features.shape[2]
            features = features.permute(0, 2, 1).view(batch_size, feat_dim, int(seq_len**0.5), int(seq_len**0.5))
        
        # Sample features: [B, C, H, W], [B, D, H, W, 2] -> [B, C, D, H, W]
        sampled_features = F.grid_sample(
            features, points_2d, mode='bilinear', align_corners=True
        )
        
        return sampled_features
        
    def forward(self, view_features, camera_params):
        """Project features from multiple views into a unified 3D representation
        
        Args:
            view_features (List[torch.Tensor]): Features from each view [B, C, H, W]
            camera_params (List[Dict]): Camera parameters for each view
            
        Returns:
            torch.Tensor: Fused 3D features [B, C, D, H, W]
        """
        batch_size = view_features[0].shape[0]
        device = view_features[0].device
        
        # Initialize volume for accumulating features
        volume_features = torch.zeros(
            batch_size, self.feature_dim, self.depth_resolution, 
            self.grid_resolution, self.grid_resolution
        ).to(device)
        
        # Initialize volume for tracking visibility/confidence
        visibility = torch.zeros(
            batch_size, 1, self.depth_resolution,
            self.grid_resolution, self.grid_resolution
        ).to(device)
        
        # Project features from each view to 3D
        for i, (features, params) in enumerate(zip(view_features, camera_params)):
            # Process 2D features
            features_2d = self.feature_projector(features)
            
            # Project to 3D volume
            volume = self.project_features_to_3d(features_2d, None, params)
            
            # Predict visibility/confidence for this view
            view_visibility = self.visibility_net(volume)
            
            # Accumulate features and visibility
            volume_features += volume * view_visibility
            visibility += view_visibility
        
        # Normalize by visibility (avoid division by zero)
        fused_volume = volume_features / (visibility + 1e-6)
        
        # Apply 3D CNN to refine features
        fused_volume = self.volume_encoder(fused_volume)
        
        return fused_volume
