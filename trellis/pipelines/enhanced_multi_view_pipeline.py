from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as T
from .trellis_image_to_3d import TrellisImageTo3DPipeline
from ..modules.camera_fusion import CameraAwareFeatureFusion
from ..modules.background_removal import create_background_remover
from ..modules.camera_pose_estimation import CameraPoseEstimator
from ..models.enhanced_sparse_structure_flow import EnhancedSparseStructureFlowModel
from ..models.enhanced_structured_latent_flow import EnhancedSLatFlowModel


class EnhancedMultiViewPipeline(TrellisImageTo3DPipeline):
    """Enhanced pipeline that extends TRELLIS for better multi-view processing."""
    def __init__(self, 
                 use_rembg=True, 
                 use_enhanced_models=True,
                 camera_feature_method='sift', 
                 **kwargs):
        super().__init__(**kwargs)
        
        # Initialize our new components
        self.pose_estimator = CameraPoseEstimator(
            feature_method=camera_feature_method,
            device=self.device
        )
        
        # Create background remover
        bg_method = 'rembg' if use_rembg else 'simple'
        self.background_remover = create_background_remover(method=bg_method, device=self.device)
        
        # Add camera-aware feature fusion module
        self.feature_fusion = CameraAwareFeatureFusion(
            feature_dim=self.image_encoder.out_dim,  # Match the dimension of image encoder
            hidden_dim=1024,
            num_heads=8,
            num_layers=3
        )
        
        # Flag to control whether we use the enhanced models
        self.use_enhanced_models = use_enhanced_models
        
        # Replace standard models with enhanced versions if requested
        if use_enhanced_models:
            self._create_enhanced_models()
            
    def _create_enhanced_models(self):
        """Replace standard models with enhanced versions"""
        # Parameters from original models
        sparse_structure_params = self.sparse_structure_sampler.model_params
        slat_params = self.slat_sampler.model_params
        
        # Create enhanced sparse structure flow model
        resolution = sparse_structure_params['resolution']
        model_channels = sparse_structure_params['model_channels']
        self.enhanced_sparse_structure_model = EnhancedSparseStructureFlowModel(
            resolution=resolution,
            in_channels=sparse_structure_params['in_channels'],
            model_channels=model_channels,
            cond_channels=sparse_structure_params['cond_channels'],
            out_channels=sparse_structure_params['out_channels'],
            num_blocks=sparse_structure_params['num_blocks'],
            num_heads=sparse_structure_params.get('num_heads'),
            num_head_channels=sparse_structure_params.get('num_head_channels', 64),
            mlp_ratio=sparse_structure_params.get('mlp_ratio', 4.0),
            use_fp16=sparse_structure_params.get('use_fp16', False),
        ).to(self.device)
        
        # Load weights from original model
        self.enhanced_sparse_structure_model.load_state_dict(
            self.sparse_structure_sampler.model.state_dict(), strict=False
        )
        
        # Create enhanced structured latent flow model
        self.enhanced_slat_model = EnhancedSLatFlowModel(
            in_channels=slat_params['in_channels'],
            cond_channels=slat_params['cond_channels'],
            hidden_channels=slat_params['hidden_channels'],
            out_channels=slat_params['out_channels'],
            num_input_blocks=slat_params.get('num_input_blocks', 3),
            num_output_blocks=slat_params.get('num_output_blocks', 3),
            num_bottleneck_blocks=slat_params.get('num_bottleneck_blocks', 6),
            skip_connections=slat_params.get('skip_connections', True),
            use_viewdir=slat_params.get('use_viewdir', False),
            use_fp16=slat_params.get('use_fp16', False),
        ).to(self.device)
        
        # Load weights from original model
        self.enhanced_slat_model.load_state_dict(
            self.slat_sampler.model.state_dict(), strict=False
        )
        
    def sample_sparse_structure(self, cond, num_samples=1, sampler_params=None, camera_params=None):
        """Sample sparse structure with enhanced model if available"""
        if not self.use_enhanced_models or not hasattr(self, 'enhanced_sparse_structure_model'):
            # Use the original implementation
            return super().sample_sparse_structure(cond, num_samples, sampler_params)
        
        # Use the enhanced model with camera parameters
        sampler_params = {} if sampler_params is None else sampler_params
        
        # Get original parameters but use our enhanced model
        result = self.sparse_structure_sampler.sample(
            cond, 
            num_samples=num_samples,
            model=self.enhanced_sparse_structure_model,  # Use enhanced model
            camera_params=camera_params,  # Pass camera parameters
            **sampler_params
        )
        
        return result
    
    def sample_slat(self, cond, coords, sampler_params=None, camera_params=None):
        """Sample structured latents with enhanced model if available"""
        if not self.use_enhanced_models or not hasattr(self, 'enhanced_slat_model'):
            # Use the original implementation
            return super().sample_slat(cond, coords, sampler_params)
        
        # Use the enhanced model with camera parameters
        sampler_params = {} if sampler_params is None else sampler_params
        
        # Get original parameters but use our enhanced model
        result = self.slat_sampler.sample(
            cond, 
            coords,
            model=self.enhanced_slat_model,  # Use enhanced model
            camera_params=camera_params,  # Pass camera parameters
            **sampler_params
        )
        
        return result
        
    def process_multi_view(self, images, view_types=None):
        """Process multiple views with background removal and camera estimation
        
        Args:
            images (List[PIL.Image]): List of input images
            view_types (List[str], optional): List of view types for each image
                If None, views will be automatically named
                
        Returns:
            Tuple[List[PIL.Image], List[np.ndarray], List[Dict]]: 
                Processed images, masks, and camera parameters
        """
        processed_images = []
        masks = []
        
        # Remove backgrounds
        for image in images:
            processed_image, mask = self.background_remover.remove_background(image)
            processed_images.append(processed_image)
            masks.append(mask)
        
        # Estimate camera parameters
        camera_params = self.pose_estimator.estimate_poses(images)
        
        # Update view types if provided
        if view_types:
            for i, view_type in enumerate(view_types):
                if i < len(camera_params):
                    camera_params[i]['view_type'] = view_type
        
        return processed_images, masks, camera_params
    
    def get_multi_view_cond(self, images, camera_params):
        """Get conditioning features from multiple images with camera fusion
        
        Args:
            images (List[PIL.Image]): List of processed images
            camera_params (List[Dict]): List of camera parameters
            
        Returns:
            torch.Tensor: Fused conditioning features
        """
        # Extract features from each image
        view_features = []
        for image in images:
            # Preprocess image as in the base class
            image_tensor = self.preprocess_image(image)
            # Get features from image encoder
            with torch.no_grad():
                features = self.image_encoder(image_tensor.to(self.device))
            view_features.append(features)
        
        # Fuse features with camera awareness
        fused_features = self.feature_fusion(view_features, camera_params)
        
        return fused_features
    
    def run_multi_view(self, images, num_samples=1, formats=None, **kwargs):
        """Run the pipeline with multiple views
        
        Args:
            images (List[PIL.Image]): List of input images
            num_samples (int): Number of samples to generate
            formats (List[str], optional): List of output formats
            **kwargs: Additional keyword arguments
                view_types (List[str], optional): List of view types for each image
                
        Returns:
            Dict: Dictionary of outputs
        """
        if formats is None:
            formats = [self.default_format]
            
        view_types = kwargs.get('view_types', None)
        
        # Process images and get camera parameters
        processed_images, masks, camera_params = self.process_multi_view(images, view_types)
        
        # Get camera-aware conditioning from processed images
        cond = self.get_multi_view_cond(processed_images, camera_params)
        
        # Sample sparse structure with camera awareness
        sparse_structure_sampler_params = kwargs.get('sparse_structure_sampler_params', {})
        coords = self.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params, camera_params)
        
        # Sample structured latents with camera awareness
        slat_sampler_params = kwargs.get('slat_sampler_params', {})
        slat = self.sample_slat(cond, coords, slat_sampler_params, camera_params)
        
        # Decode to output formats
        outputs = self.decode_slat(slat, formats)
        
        # Add camera parameters and masks to outputs for reference
        outputs['camera_params'] = camera_params
        outputs['masks'] = masks
        outputs['processed_images'] = processed_images
        
        return outputs
